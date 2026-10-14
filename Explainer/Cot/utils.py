import pickle
import json

import numpy as np
import torch
import os
from pathlib import Path

from arguments import args

import math
import re
import csv
from collections import Counter, defaultdict
from fractions import Fraction

def code_to_class(code):
    if len(code)<=4:
        return code
    else:
        s1 = code[:4]
        s2 = code[4:8]
        s3 = code[8:]

        s2 = str(int(s2))
        for _ in range(4):
            if s3 and s3[-1] == '0':
                s3 = s3[:-1]

        return s1+s2+"/"+s3
def extract_mask_metapaths_ids(edge_index):
    result_name = edge_index
    result_dir = f'new_edges_{result_name}'
    mask_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\model/{result_dir}'
    mask_logit_result_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\mask_logit_result/{result_dir}'

    # 提取出mask
    with open(f'{mask_dir}/metapath_masks.pkl', 'rb') as f:
        metapath_masks = pickle.load(f)
    with open(f'{mask_logit_result_dir}/metapath_mask_logit_result.pkl', 'rb') as file:
        # 加载testipc抽出的权重，为数据框列表，每个数据框是各类型元路径的权重
        metapath_mask_logit_list = pickle.load(file)

    metapath_masks_list = []
    for mtype,metapath_mask in metapath_masks.items():
        metapath_mask = torch.where(metapath_mask > 0.5, torch.ones_like(metapath_mask), torch.zeros_like(metapath_mask))
        mask_np = metapath_mask.cpu().numpy().flatten()
        metapath_masks_list.append(mask_np)

    mask_metapaths = []
    for idx in range(len(metapath_masks_list)):
        selected_rows = metapath_mask_logit_list[idx][metapath_masks_list[idx] > 0]
        mask_metapaths.append(selected_rows)
    return mask_metapaths

def trans_mtype_(mtype,mtype_metapaths:list,id2ipc,id2patent,id2paper,num_ipc, num_patent):
    trans_result = []
    if mtype == 0:
        for metapath in mtype_metapaths:
            if metapath[0]!=metapath[2]:
                metapath_trans = []
                metapath_trans.append(id2ipc[str(metapath[0])])
                metapath_trans.append(id2patent[str(metapath[1]-num_ipc)])
                metapath_trans.append(id2ipc[str(metapath[2])])
                trans_result.append(metapath_trans)
            else:
                continue

    if mtype == 1:
        for metapath in mtype_metapaths:
            if metapath[0] != metapath[3] and metapath[1] != metapath[2]:
                metapath_trans = []
                metapath_trans.append(id2ipc[str(metapath[0])])
                metapath_trans.append(id2patent[str(metapath[1]-num_ipc)])
                metapath_trans.append(id2patent[str(metapath[2]-num_ipc)])
                metapath_trans.append(id2ipc[str(metapath[3])])
                trans_result.append(metapath_trans)
            else:
                continue

    if mtype == 2:
        for metapath in mtype_metapaths:
            if metapath[0] != metapath[4] and metapath[1] != metapath[3]:
                metapath_trans = []
                metapath_trans.append(id2ipc[str(metapath[0])])
                metapath_trans.append(id2patent[str(metapath[1]-num_ipc)])
                metapath_trans.append(id2paper[str(metapath[2]-num_ipc-num_patent)])
                metapath_trans.append(id2patent[str(metapath[3]-num_ipc)])
                metapath_trans.append(id2ipc[str(metapath[4])])
                trans_result.append(metapath_trans)
            else:
                continue
    return trans_result

def trans_mask_metapaths(mask_metapaths_list,id2ipc,id2patent,id2paper,num_ipc, num_patent):
    result_mask_metapaths = []
    for mtype in range(args.num_metapath_types):
        mtype_metapaths = mask_metapaths_list[mtype][f'metapath_mtype{mtype}'].tolist()
        trans_mtype_metapaths = trans_mtype_(mtype,mtype_metapaths,id2ipc,id2patent,id2paper,num_ipc, num_patent)
        result_mask_metapaths.append(trans_mtype_metapaths)
    return result_mask_metapaths

def get_metapath_explanation(edge_index, id2ipc, id2patent, id2paper):
    num_ipc = args.num_ipc
    num_patent = args.num_patent
    mask_metapaths = extract_mask_metapaths_ids(edge_index)
    mask_metapaths_converted = trans_mask_metapaths(mask_metapaths, id2ipc, id2patent, id2paper, num_ipc, num_patent)
    return mask_metapaths_converted

def get_metapath_from_CN(CN_num,metapath_explanation):
    patent_list = []
    paper_list = []
    for mtype in range(args.num_metapath_types):
        if len(metapath_explanation[mtype])>0:
            if mtype == 0:
                for metapath in metapath_explanation[mtype]:
                    if metapath[0]==CN_num or metapath[-1]==CN_num:
                        patent_list.append(metapath[1])
            elif mtype == 1:
                for metapath in metapath_explanation[mtype]:
                    if metapath[0] == CN_num or metapath[-1] == CN_num:
                        patent_list.append(metapath[1])
                        patent_list.append(metapath[2])
            elif mtype == 2:
                for metapath in metapath_explanation[mtype]:
                    if metapath[0] == CN_num or metapath[-1] == CN_num:
                        patent_list.append(metapath[1])
                        patent_list.append(metapath[3])
                        paper_list.append(metapath[2])
    return {"patent_list":list(set(patent_list)),"paper_list":list(set(paper_list))}

def get_explained_edges(folder_path):
    subfolders = []  # 创建一个空列表来存储子文件夹的名字
    for entry in os.scandir(folder_path):  # 使用os.scandir()遍历文件夹
        if entry.is_dir():  # 检查条目是否为文件夹
            subfolders.append(entry.name)  # 如果是文件夹，则将其名字添加到列表中

    explained_edges_index = [int(s.split("_")[-1]) for s in subfolders]
    return explained_edges_index

def get_ipc_info(ipc:str,ipc_tree):
    ipc_path = list(ipc_tree.rsearch(ipc))
    result = ""
    for i in reversed(ipc_path):
        node = ipc_tree.get_node(i)
        node_data = node.data
        result += node_data + " - "
    result = result[:-3]
    return result

def get_ipc_info_from_all_trees(ipc:str,ipc_trees):
    for year in range(2023, 2009, -1):
        try:
            result = get_ipc_info(ipc,ipc_trees[year][ipc[0]])
            return result
        except Exception as e:
            continue

def get_paper_info(paper_title,paper_data):
    paper_row = paper_data[paper_data['Title'] == paper_title]
    result = {}

    title = paper_row['Title_en'].iloc[0]
    abstract = paper_row['Summary_en'].iloc[0]
    citations = paper_row['Citations'].iloc[0]

    result['title'] = title
    result['abstract'] = abstract
    result['citations'] = citations
    return result

def get_patent_info(publication_num, patent_data):
    patent_row = patent_data[patent_data['公开（公告）号'] == publication_num]
    assert len(patent_row) == 1
    result = {}

    title = patent_row['标题 (英文)'].iloc[0]
    abstract = patent_row['摘要 (英文)'].iloc[0]
    citations = patent_row['被引证次数'].iloc[0]
    claims = patent_row['权利要求数量'].iloc[0]
    tech_advancement = patent_row['技术先进性'].iloc[0]
    tech_efficacy = patent_row['技术功效句（英文）'].iloc[0]

    result['title'] = title
    result['abstract'] = abstract
    result['citations'] = citations
    result['claims'] = claims
    result['tech_advancement'] = tech_advancement
    result['tech_efficacy'] = tech_efficacy

    return result

def get_citation_relation(patent1,patent2,patent_data):
    patent1_row = patent_data[patent_data['公开（公告）号'] == patent1]
    patent2_row = patent_data[patent_data['公开（公告）号'] == patent2]

    patent1_citations = patent1_row['引证专利'].iloc[0]
    patent2_citations = patent2_row['引证专利'].iloc[0]

    result = ""
    if not isinstance(patent1_citations, str):
        if np.isnan(patent1_citations):
            patent1_citations = ""
    if not isinstance(patent2_citations, str):
        if np.isnan(patent2_citations):
            patent2_citations = ""

    if patent2 in patent1_citations:
        result = f"patent {patent1} cites patent {patent2}"
    elif patent1 in patent2_citations:
        result = f"patent {patent2} cites patent {patent1}"

    assert len(result) > 0
    return result

def get_metapath_desc(metapaths,patent_data,paper_data):
    num_metapath_types = args.num_metapath_types
    desc_list = []
    for m_type in range(num_metapath_types):
        m_type_metapaths = metapaths[m_type]
        if len(m_type_metapaths) != 0:
            if m_type == 0:
                for metapath in m_type_metapaths:
                    desc = f"{code_to_class(metapath[0])} and {code_to_class(metapath[2])} are the IPC classification numbers of patent {metapath[1]}."
                    desc_list.append(desc)
            if m_type == 1:
                for metapath in m_type_metapaths:
                    desc = f"{code_to_class(metapath[0])} is the IPC classification number of patent {metapath[1]}, and {code_to_class(metapath[3])} is the IPC classification number of patent {metapath[2]}. In addition, {get_citation_relation(metapath[1],metapath[2],patent_data)}."
                    desc_list.append(desc)
            if m_type == 2:
                for metapath in m_type_metapaths:
                    desc = f"{code_to_class(metapath[0])} is the IPC classification number of patent {metapath[1]}, and {code_to_class(metapath[4])} is the IPC classification number of patent {metapath[3]}. In addition, both patent {metapath[1]} and patent {metapath[3]} cite paper '{get_paper_info(metapath[2],paper_data)['title']}'."
                    desc_list.append(desc)

    metapath_num = len(desc_list)
    result = ""
    for i in range(metapath_num):
        matapath_desc = f"Metapath {i+1}: {desc_list[i]}\n"
        result += matapath_desc
    return result

def get_metapath_desc_fromCN(CN,metapath_from_CN,patent_data,paper_data):
    result = ""
    patent_list = metapath_from_CN['patent_list']
    paper_list = metapath_from_CN['paper_list']
    if len(patent_list)>0:
        CN_patent_desc = get_patent_desc_fromCN(CN,metapath_from_CN,patent_data)
        result += CN_patent_desc
    if len(paper_list)>0:
        CN_paper_desc = get_paper_desc_fromCN(CN,metapath_from_CN,paper_data)
        result += "\n"+CN_paper_desc
    return result


def get_paper_desc(metapaths,paper_data):
    paper_metapaths = metapaths[2]
    paper_list = []
    result = ""
    if len(paper_metapaths) == 0:
        return result
    else:
        for metapath in paper_metapaths:
            paper_list.append(metapath[2])

        paper_list = list(set(paper_list))
        for i in range(len(paper_list)):
            paper = paper_list[i]
            paper_info = get_paper_info(paper,paper_data)
            paper_desc = f"Paper '{paper}': {paper_info}\n"
            result += paper_desc
        return result

def get_paper_desc_fromCN(CN,metapath_from_CN,paper_data):
    result = f"Here are the papers in the metapaths related to {CN}:\n"
    paper_list = list(set(metapath_from_CN["paper_list"]))
    for i in range(len(paper_list)):
        paper = paper_list[i]
        paper_info_ = get_paper_info(paper, paper_data)
        keys_to_extract=['title','abstract']
        paper_info = {key: paper_info_.get(key) for key in keys_to_extract}
        paper_desc = f"Paper '{paper}': {paper_info}\n"
        result += paper_desc
    return result

def get_patent_desc(metapaths,patent_data):
    num_metapath_types = args.num_metapath_types
    patent_list = []
    result = ""

    for m_type in range(num_metapath_types):
        m_type_metapaths = metapaths[m_type]
        if len(m_type_metapaths) != 0:
            if m_type == 0:
                for metapath in m_type_metapaths:
                    patent_list.append(metapath[1])
            if m_type == 1:
                for metapath in m_type_metapaths:
                    patent_list.append(metapath[1])
                    patent_list.append(metapath[2])
            if m_type == 2:
                for metapath in m_type_metapaths:
                    patent_list.append(metapath[1])
                    patent_list.append(metapath[3])

    patent_list = list(set(patent_list))
    for i in range(len(patent_list)):
        patent = patent_list[i]
        patent_info = get_patent_info(patent,patent_data)
        patent_desc = f"Patent {patent}: {patent_info}\n"
        result += patent_desc
    return result

def get_patent_desc_fromCN(CN,metapath_from_CN,patent_data):
    result = f"Here are the patents in the metapaths related to {CN}:\n"
    patent_list = list(set(metapath_from_CN["patent_list"]))
    for i in range(len(patent_list)):
        patent = patent_list[i]
        patent_info_ = get_patent_info(patent, patent_data)
        keys_to_extract=['title','abstract']
        patent_info = {key: patent_info_.get(key) for key in keys_to_extract}
        patent_desc = f"Patent '{patent}': {patent_info}\n"
        result += patent_desc
    return result

def get_ipc_desc(target_edge,metapaths,all_ipc_trees):
    num_metapath_types = args.num_metapath_types
    ipc_list = list(target_edge)
    result = ""

    for m_type in range(num_metapath_types):
        m_type_metapaths = metapaths[m_type]
        if len(m_type_metapaths) != 0:
            if m_type == 0:
                for metapath in m_type_metapaths:
                    ipc_list.append(metapath[0])
                    ipc_list.append(metapath[2])
            if m_type == 1:
                for metapath in m_type_metapaths:
                    ipc_list.append(metapath[0])
                    ipc_list.append(metapath[3])
            if m_type == 2:
                for metapath in m_type_metapaths:
                    ipc_list.append(metapath[0])
                    ipc_list.append(metapath[4])

    ipc_list = list(set(ipc_list))
    for i in range(len(ipc_list)):
        ipc = ipc_list[i]
        for year in range(2023, 2009, -1):
            ipc_trees = all_ipc_trees[year]
            ipc_tree = ipc_trees[ipc[0]]
            if ipc_tree.contains(ipc):
                ipc_path = list(ipc_tree.rsearch(ipc))
                ipc_desc = f"IPC {code_to_class(ipc)}: "
                for path_i in reversed(ipc_path):
                    node = ipc_tree.get_node(path_i)
                    node_data = node.data
                    ipc_desc += node_data + " - "
                result += ipc_desc[:-3] + ".\n"
                break
            else:
                continue
    return result

INVALID_ANS = "[invalid]"

def load_data(frn, format="list", mode="r"):
    """Load data from a file.
    :param frn (str): The dataset file name.

    :return: The dataset (a list of examples, each as a dictionary).
    """
    if not os.path.exists(frn):
        filename = Path(frn)
        filename.parent.mkdir(parents=True, exist_ok=True)
        return []
    if frn.endswith(".jsonl") or frn.endswith(".txt"):
        with open(frn, mode) as fr:
            rtns = [] if format == "list" else {}
            for i, line in enumerate(fr):
                if line.strip() == "":
                    continue
                try:
                    line = json.loads(line)
                    if format == "list":
                        rtns.append(line)
                    else:
                        idx = line["idx"]
                        rtns[idx] = line
                except json.decoder.JSONDecodeError as e:
                    print(f"Error in line {i}: {line}\n {e}")
                    exit(-1)
        return rtns
    elif frn.endswith(".json"):
        with open(frn, mode) as fr:
            rtns = json.load(fr)
        return rtns
