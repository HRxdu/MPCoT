import json
import pickle

import pandas as pd
import numpy as np
from openai import OpenAI
from treelib import Tree, Node
import torch
import dgl
import os
from langchain_community.chat_models import ChatOpenAI
import random

from arguments import args
from Cot.utils import *
from Cot.models import *
from Cot.autodl import autodlEnv

if __name__ == '__main__':
    toa_model_dir = args.toa_model_dir
    data_dir = 'result/{}/'.format(args.dataset)

    with open(f'data/2023/ipc_tree/ipc_trees.pkl', 'rb') as file:
        ipc_trees = pickle.load(file)
    with open(f'data/all_ipc_trees.pkl', 'rb') as file:
        all_ipc_trees = pickle.load(file)
    paper_data = pd.read_excel("data/translated_data/translated_papers.xlsx")
    patent_data = pd.read_excel("data/translated_data/translated_patents_2.xlsx")

    # 导入预测的新边
    """with open(toa_model_dir + '/result/大模型/static_new_edges.pkl', 'rb') as file:
        new_edges = pickle.load(file)"""
    with open('output/new_edges_test_TP.pkl', 'rb') as file:
        new_edges = pickle.load(file)

    id2ipc = json.load(open(toa_model_dir + '/' + data_dir + 'id2ipc.json', 'r'))
    id2patent = json.load(open(toa_model_dir + '/' + data_dir + 'id2patent.json', 'r'))
    id2paper = json.load(open(toa_model_dir + '/' + data_dir + 'id2paper.json', 'r'))

    for idx,e in enumerate(new_edges):
        trans_e = (id2ipc[f'{e[0].item()}'],id2ipc[f'{e[1].item()}'])
        new_edges[idx] = trans_e

    explained_edges = get_explained_edges('output/mask_logit_result')

    random.seed(42)
    numbers = list(range(len(explained_edges)))
    random.shuffle(numbers)

    gen_count = 100  #生成解释的数量,随机抽样

    count = 0
    for random_idx in numbers:
        target_edge_idx = explained_edges[random_idx]
        #print(f"第{count}个解释开始处理:(0-{gen_count-1})")
        #count += 1
    #for target_edge_idx in explained_edges[0:1]:
        # target_edge_idx = 100
        target_edge = new_edges[target_edge_idx]
        metapath_explanation = get_metapath_explanation(target_edge_idx, id2ipc, id2patent, id2paper)
        target_edge_processed = list(map(code_to_class, target_edge))

        CN1_num = target_edge[0]
        CN2_num = target_edge[1]

        metapath_from_CN1 = get_metapath_from_CN(CN1_num, metapath_explanation)
        metapath_from_CN2 = get_metapath_from_CN(CN2_num, metapath_explanation)

        CN1 = target_edge_processed[0]
        CN2 = target_edge_processed[1]

        ipc_info = {CN1: get_ipc_info_from_all_trees(CN1_num, all_ipc_trees),
                    CN2: get_ipc_info_from_all_trees(CN2_num, all_ipc_trees)}

        CN1_metapath_desc = get_metapath_desc_fromCN(CN1,metapath_from_CN1,patent_data,paper_data)
        CN2_metapath_desc = get_metapath_desc_fromCN(CN2, metapath_from_CN2, patent_data, paper_data)

        metapath_desc = {
            0:CN1_metapath_desc,
            1:CN2_metapath_desc,
            2:CN1_metapath_desc+"\n"+CN2_metapath_desc
        }

        """////////////////////////////////////////"""
        total_length = 0
        for value in metapath_desc.values():
            total_length += len(value)
        if total_length > 10:
            print(f"第{count}个解释开始处理:(0-{gen_count - 1})")
            count += 1
        else:
            continue

        if count>99:
            break
        """////////////////////////////////////////"""


        data = {
            "ipc_pair": target_edge_processed,
            "CN1":target_edge_processed[0],
            "CN2": target_edge_processed[1],
            "ipc_info": ipc_info,
            "CN1_metapath_desc":CN1_metapath_desc,
            "CN2_metapath_desc":CN2_metapath_desc,
            "metapath_desc":metapath_desc,
            "idx": target_edge_idx

        }

        chat = ChatOpenAI(
            #base_url="http://localhost:6006/v1",
            #base_url="http://localhost:1234/v1",
            base_url="https://api.siliconflow.cn/v1",
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            openai_api_key=args.api_key
        )
        #chat = autodlEnv()
        env = LogiCoTEnv(
            chat=chat,
            max_steps=args.max_steps,
            output="output/lot/output_dev.jsonl",
            debug=args.debug,
            parallel_argument_num=args.parallel_argument_num,
            refined_tolerance=args.refined_tolerance
        )
        agent = LogiAgent(
            chat=chat,
            check_refined=True,
            mode=args.agent_mode,
        )

        state, terminate = env.reset(data) #state包含当前节点的信息，包括self.sys_msg, col（节点idx）, P（节点的thought）, refined(是否修正了)
        for _ in range(100):
            if terminate:
                #metrics.update(env.data)
                #metrics.log()
                print(f"edge_dix:{target_edge_idx}处理完成！")
                break
            try:
                action = agent.policy(state) #对当前节点进行验证，是否通过，action包括advice:步骤col的内容，passed:是否通过
                state, terminate = env.step(action)
            except Exception as e:
                env.recover(e)
                terminate = True

