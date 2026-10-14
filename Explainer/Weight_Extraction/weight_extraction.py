import sys
import os
import pandas as pd
import json
import pickle

import torch
import dgl
import netron
from torchviz import make_dot
import dgl.function as fn

static_model_path= r'D:/Pycharm_project/interpretable_TOA/static_mode/TOA_model'
# 添加到 sys.path
if static_model_path not in sys.path:
    sys.path.append(static_model_path)

# 导入自定义包
from myModel import MTHG
from arguments import args
from process.ipc_adj_matrix import ipc_transform
from utils.util import *


"""def message_passing(edges):
    ft = edges.data['efeature'] * edges.data['weight']
    return {'ft': ft}
"""
def Weight_Extract(target_ipc:list,result_path:str):
    """
    parameter:
    target_ipc:the ipc list you want to extract the metapath weight,Example format is ['G06F0016215000', 'G16H0010000000']
    result_path: the path the result saved
    result_name: the name of the result you want to save as
    """
    metapath_Weight_Extracted_DIR ='D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/metapath_Weight_Extracted'
    intermediate_variable_DIR = metapath_Weight_Extracted_DIR+'/intermediate variable'

    if not os.path.isdir(metapath_Weight_Extracted_DIR):
        os.mkdir(metapath_Weight_Extracted_DIR)
    if not os.path.isdir(intermediate_variable_DIR):
        os.mkdir(intermediate_variable_DIR)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # Assumes a saved base model as input and model name to get the right directory.
    data_dir =static_model_path+ '/result/{}/'.format(args.dataset)
    output_dir = data_dir + 'model_output/'

    # Load graphs and features.
    device = args.device
    graph = dgl.load_graphs(args.graph_path)[0][0]
    feature_dict = graph.ndata['f'] # keys(['ipc', 'paper', 'patent'])

    mode = 'eval'
    model = MTHG(feature_dict, args,feature_extract = True).to(device)
    model.load_state_dict(torch.load(output_dir + 'model/checkpoint.pt'))
    model.eval()

    ipc2id = json.load(open(data_dir + 'ipc2id.json', 'r'))
    id2ipc = json.load(open(data_dir + 'id2ipc.json', 'r'))

    target_ipc = torch.tensor(target_ipc)

    target_ipc_list = target_ipc.numpy().tolist()
    idx2ipc = dict(zip(list(range(len(target_ipc_list))), target_ipc_list))

    metapaths_till_t = get_metapaths(target_ipc, timestamp=args.test_timestamp)
    metapath_graphs_till_t = get_metapath_graphs(metapaths_till_t, timestamp=args.test_timestamp)

    embd = model([target_ipc, metapaths_till_t, metapath_graphs_till_t, args.test_timestamp, mode])

    metapath_weight_list=[]
    #metapath_type_feature_list=[]

    for mtype in range(args.num_metapath_types):
        #导入导出的权重及元路径实例特征
        with open(intermediate_variable_DIR + f'/a_drop_mtype{mtype}.pkl', 'rb') as file:
            a_drop_mtype = pickle.load(file)
        with open(intermediate_variable_DIR + f'/attn_gate_mtype{mtype}.pkl', 'rb') as file:
            attn_gate_mtype = pickle.load(file)
        with open(intermediate_variable_DIR + '/beta.pkl', 'rb') as file:
            beta = pickle.load(file)

        metapath_weight=a_drop_mtype * attn_gate_mtype * beta[mtype]
        metapath_weight_list.append(metapath_weight)
        #metapath_type_feature_list.append(result)

    metapath_weight_list = [torch.sum(type, dim=1) for type in metapath_weight_list]

    df_list=[]

    for mtype in range(args.num_metapath_types):
        weight_type=metapath_weight_list[mtype].detach().cpu()
        metapath_type=metapaths_till_t[mtype].detach().cpu()

        # 将张量转换为列表
        weights = weight_type.numpy().flatten().tolist()
        metapaths = metapath_type.numpy().tolist()

        # 创建DataFrame
        df = pd.DataFrame({
            f'weight_mtype{mtype}': weights,
            f'metapath_mtype{mtype}': metapaths
        })
        df_list.append(df)

    with open(f'{result_path}/weight_extracted.pkl', 'wb') as file:
        pickle.dump(df_list, file)

    return df_list

def weight_sort(weight_list):
    result_df = pd.DataFrame(columns=['weight','metapath'])

    for df in weight_list:
        column_names = df.columns
        new_names = {f'{column_names[0]}': 'weight', f'{column_names[1]}': 'metapath'}
        df.rename(columns=new_names, inplace=True)

    for df in weight_list:
        result_df = pd.concat([result_df, df], axis=0, ignore_index=True)

    result_df = result_df.sort_values(by='weight', ascending=False)

    return result_df


