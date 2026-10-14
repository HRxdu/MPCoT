import dgl
import torch
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

from arguments import args
from dgl.dataloading.negative_sampler import GlobalUniform

with open(args.metapath_path, 'rb') as f:
    metapath_dict = pickle.load(f)
    f.close()

def rand_normal(original_data):
    # 正态分布
    seed = 42
    gen = torch.Generator(device=original_data.device)  # 指定生成器的设备（CPU/GPU）
    gen.manual_seed(seed)
    mean = original_data.mean(dim=None)
    std = original_data.std(dim=None)
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        mean = mean.item()
        std = std.item()
    random_tensor = torch.normal(mean=mean, std=std, size=original_data.shape,device=original_data.device,generator=gen)
    return random_tensor

def newedge_ipc_trans(id_tuple,id2ipc):
    # 元组 (a, b) 包含两个ID
    a, b = id_tuple

    # 使用字典映射ID到IPC
    ipc_a = id2ipc[f'{a}']
    ipc_b = id2ipc[f'{b}']

    # 创建包含IPC的元组
    ipc_tuple = (ipc_a, ipc_b)
    return ipc_tuple

def edge_filter4now(edges, t):
    return edges.data['t'] == t

def edge_filter4not_now(edges, t):
    return edges.data['t'] != t

def get_train_graph(graph_path=args.graph_path, timestamp=args.test_timestamp):
    graph = dgl.load_graphs(graph_path)[0][0]
    graph.remove_edges(graph.filter_edges(lambda x: edge_filter4now(x, timestamp), etype='co-occurs with'), etype='co-occurs with')
    result = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])
    #result = dgl.remove_self_loop(result)
    result = dgl.to_bidirected(result)

    # 获取'co-occurs with'类型的边,获取其中的单向边
    bi_co_occurs_edges = list(result.edges(etype='co-occurs with'))
    bi_combined = list(zip(bi_co_occurs_edges[0].numpy(), bi_co_occurs_edges[1].numpy()))
    un_co_occurs_edges = []
    for e in bi_combined:
        if e not in un_co_occurs_edges and (e[1], e[0]) not in un_co_occurs_edges:
            un_co_occurs_edges.append(e)

    # 转换为边的ID
    eids = torch.tensor([result.edge_id(src, dst) for src, dst in un_co_occurs_edges])
    # 使用remove_edges方法移除特定边
    result = dgl.remove_edges(result, eids)

    return result

def get_test_graph(graph_path=args.graph_path, timestamp=args.test_timestamp):
    graph = dgl.load_graphs(graph_path)[0][0]
    graph.remove_edges(graph.filter_edges(lambda x: edge_filter4not_now(x, timestamp), etype='co-occurs with'), etype='co-occurs with')
    result = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])
    #result = dgl.remove_self_loop(result)
    result = dgl.to_bidirected(result)

    # 获取'co-occurs with'类型的边,获取其中的单向边
    bi_co_occurs_edges = list(result.edges(etype='co-occurs with'))
    bi_combined = list(zip(bi_co_occurs_edges[0].numpy(), bi_co_occurs_edges[1].numpy()))
    un_co_occurs_edges = []
    for e in bi_combined:
        if e not in un_co_occurs_edges and (e[1], e[0]) not in un_co_occurs_edges:
            un_co_occurs_edges.append(e)

    # 转换为边的ID
    eids = torch.tensor([result.edge_id(src, dst) for src, dst in un_co_occurs_edges])
    # 使用remove_edges方法移除特定边
    result = dgl.remove_edges(result, eids)

    return result

def get_static_train_test_graph(graph_path=args.graph_path,timestamp=args.test_timestamp):
    g = dgl.load_graphs(graph_path)[0][0]
    etype = 'co-occurs with'
    g = dgl.edge_type_subgraph(g, etypes=[etype])
    g = dgl.remove_self_loop(g)
    g = dgl.to_bidirected(g)

    # 获取'co-occurs with'类型的边,获取其中的单向边
    bi_co_occurs_edges = list(g.edges(etype=etype))
    bi_combined = list(zip(bi_co_occurs_edges[0].numpy(), bi_co_occurs_edges[1].numpy()))
    un_co_occurs_edges = []
    for e in bi_combined:
        if e not in un_co_occurs_edges and (e[1],e[0]) not in un_co_occurs_edges:
            un_co_occurs_edges.append(e)

    # 转换为边的ID
    eids = torch.tensor([g.edge_id(src, dst) for src, dst in un_co_occurs_edges])
    # 使用remove_edges方法移除特定边
    g = dgl.remove_edges(g, eids)

    co_occurs_edges = g.edges(etype=etype)
    co_occurs_edges = list(co_occurs_edges)

    # 计算需要划分的边的数量
    num_edges = co_occurs_edges[0].shape[0]
    # 生成边的索引列表
    edge_indices = torch.arange(num_edges)

    # 使用train_test_split按8:2比例划分边
    train_indices, test_indices = train_test_split(edge_indices.numpy(), test_size=0.2)

    # 将numpy数组转回torch tensor
    train_indices = torch.tensor(train_indices)
    test_indices = torch.tensor(test_indices)

    # 创建训练图
    train_graph = dgl.remove_edges(g, test_indices, etype=etype)
    # 创建测试图
    test_graph = dgl.remove_edges(g, train_indices, etype=etype)

    result_train_graph = dgl.edge_type_subgraph(train_graph, etypes=[etype])
    result_test_graph = dgl.edge_type_subgraph(test_graph, etypes=[etype])

    return result_train_graph , result_test_graph
def get_node_pairs(graphs, nodes: torch.Tensor):
    neg_sampler = GlobalUniform(k=2)
    all_nodes = nodes.cpu().numpy().tolist()
    #result = {}
    #for tidx, t in enumerate(args.time_range):
    result_t = {}
    graph_t = graphs[0]
    graph_t = dgl.add_self_loop(graph_t)
    pos_edges = graph_t.out_edges(nodes)
    pos_nodes = pos_edges[1].numpy().tolist()
    num_pos = len(pos_nodes)

    u, v = graphs[1].edges()
    graph_t.add_edges(u, v)
    graph_t = dgl.to_bidirected(graph_t)

    neg_edges = neg_sampler(graph_t, eids=graph_t.edge_id(u=pos_edges[0], v=pos_edges[1]))
    neg_edges = tuple(map(lambda x: x[:num_pos], neg_edges))
    neg_nodes = neg_edges[0].numpy().tolist() + neg_edges[1].numpy().tolist()
    all_nodes += pos_nodes + neg_nodes
    result_t['pos'] = pos_edges
    result_t['neg'] = neg_edges
        #result[t] = result_t
    all_nodes = list(set(all_nodes))
    all_nodes = torch.tensor(all_nodes, dtype=torch.long, device=args.device)
    return all_nodes, result_t

def get_metapaths(nodes: torch.Tensor, timestamp: int, metapath_dict: dict=metapath_dict, args=args):
    '''

    :param nodes:
    :param timestamp:
    :param metapath_dict:
    :return: num_timestamps * [num_mtypes * [ Tensor.shape=[num_indices, mtype_len] ] ]
    最后获得：#所有node三种元路径，格式：
    '''
    result = []
    for mtype in range(args.num_metapath_types):
        result_mtype = []
        for node in nodes:
            node = int(node.cpu().numpy())
            result_node=[]
            for _ in range(args.min_timestamp, timestamp+1):
                metapaths_t = metapath_dict[node][_]
                metapaths_t_mtype = metapaths_t[mtype].numpy().tolist()
                result_node += metapaths_t_mtype
            if all(sublist == result_node[0] for sublist in result_node[1:]):
                result_node = [result_node[0]]
            else:
                # 使用列表推导式来创建一个不包含所有元素都是value_to_check的新列表
                result_node = [sublist for sublist in result_node if not all(element == node for element in sublist)]
            result_mtype += result_node
            #result_mtype = [sublist for sublist in result_mtype if sublist[0] != sublist[-1]]
        result.append(torch.tensor(result_mtype, device=args.device))

    """result = []
    for _ in range(args.min_timestamp, timestamp+1):
        result_tmp = []  #所有node 在t年三种元路径
        for mtype in range(args.num_metapath_types):
            result_mtype = [] #所有node 在t年mtype的元路径
            for node in nodes:
                node = int(node.cpu().numpy())
                metapaths_t = metapath_dict[node][_] 
                metapaths_t_mtype = metapaths_t[mtype].numpy().tolist()
                result_mtype += metapaths_t_mtype
            result_tmp.append(torch.tensor(result_mtype, device=args.device))
        result.append(result_tmp)"""
    return result

def get_metapath_graphs(metapaths, timestamp: int, args=args):
    #result = []
    '''metapath_dict = {
        'similar patent-cites': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'patent')],
        'similar patent-cited': [('ipc', 'assigned to', 'patent'), ('patent', 'cited by', 'patent')],
        'similar paper': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'paper')]
    }'''
    # transform = dgl.AddMetaPaths(metapath_dict, keep_orig_edges=True)
    # new_graph = transform(graph)
    ##for t in range(timestamp - args.min_timestamp + 1):
    result_tmp = []
    metapath_indices_t = metapaths
    for mtype in range(args.num_metapath_types):
        metapath_indices_t_mtype = metapath_indices_t[mtype]
        if mtype == 0:
            # i2patent2i
            ipc_src = metapath_indices_t_mtype[:, 0]
            ipc_dst = metapath_indices_t_mtype[:, 2]
            graph_tmp = dgl.DGLGraph().to(args.device)
            graph_tmp.add_nodes(args.num_ipc)
            graph_tmp.add_edges(u=ipc_dst, v=ipc_src)
            # ipcs = list(set(torch.cat([ipc_src, ipc_dst]).cpu().numpy().tolist()))
            # graph_tmp = dgl.node_subgraph(new_graph, nodes={'ipc': torch.tensor(ipcs)})
            # graph_tmp = dgl.edge_type_subgraph(graph_tmp, etypes=['co-occur']).to(args.device)
            result_tmp.append(graph_tmp)

        '''if mtype == 1:
            # i2patent2patent
            src = metapath_indices_t_mtype[:, 0]
            dst = metapath_indices_t_mtype[:, 2]
            graph_tmp = dgl.DGLGraph().to(args.device)
            graph_tmp.add_nodes(args.num_ipc + args.num_patent)
            graph_tmp.add_edges(u=dst, v=src)
            result_tmp.append(graph_tmp)'''

        if mtype == 1:
            # i2patent2patent2i
            src = metapath_indices_t_mtype[:, 0]
            dst = metapath_indices_t_mtype[:, 3]
            graph_tmp = dgl.DGLGraph().to(args.device)
            graph_tmp.add_nodes(args.num_ipc + args.num_patent)
            graph_tmp.add_edges(u=dst, v=src)
            result_tmp.append(graph_tmp)

        '''if mtype == 2:
            # i2patent2paper
            src = metapath_indices_t_mtype[:, 0]
            dst = metapath_indices_t_mtype[:, 2]
            graph_tmp = dgl.heterograph({
                ('paper', 'to', 'ipc'): (dst, src)
            }).to(args.device)
            result_tmp.append(graph_tmp)'''

        if mtype == 2:
            # i2patent2paper2patent2ipc
            src = metapath_indices_t_mtype[:, 0]
            dst = metapath_indices_t_mtype[:, 4]
            graph_tmp = dgl.DGLGraph().to(args.device)
            graph_tmp.add_nodes(args.num_ipc + args.num_patent + args.num_paper)
            graph_tmp.add_edges(u=dst, v=src)
            result_tmp.append(graph_tmp)
        ##result.append(result_tmp)
    return result_tmp

def get_metapath_timestamps_v2(metapaths, graph, num_ipc, num_patent):
    # 原代码中device未使用，保留以兼容原逻辑
    device = metapaths[0].device if metapaths else None

    def build_edge_time_mapping(g, etype):
        """
        优化：保留pandas复合索引（C实现），放弃字典映射，支持向量化查询
        返回：带MultiIndex的pandas Series (u, v) → t（去重后，保留最后一个值）
        """
        # 获取边的u, v和时间t
        u, v = g.edges(etype=etype)
        t = g.edges[etype].data['t']

        # 直接转为numpy数组（减少中间变量，指定数据类型减小内存）
        u_np = u.cpu().numpy().astype(np.int64)
        v_np = v.cpu().numpy().astype(np.int64)
        t_np = t.cpu().numpy().astype(np.int32)

        # 生成反转的边（u和v互换，时间不变）
        u_rev = v_np
        v_rev = u_np
        t_rev = t_np

        # 拼接原边和反转边（直接用numpy拼接，比pandas concat更快）
        u_all = np.concatenate([u_np, u_rev])
        v_all = np.concatenate([v_np, v_rev])
        t_all = np.concatenate([t_np, t_rev])

        # 构建复合索引的Series，去重（保留最后一个重复项，与原逻辑一致）
        # 直接用MultiIndex.from_arrays，比先建DataFrame再设索引更快
        index = pd.MultiIndex.from_arrays([u_all, v_all], names=['u', 'v'])
        time_series = pd.Series(t_all, index=index)
        # 去重：保留最后一个重复的(u, v)对
        time_series = time_series[~time_series.index.duplicated(keep='last')]

        return time_series

    def batch_query_time(meta_u, meta_v, time_series, default_time=-1):
        """
        优化：全量向量化查询（C实现的pandas索引查找），移除批次循环
        输入：tensor类型的u和v
        输出：与输入同设备、同长度的时间tensor
        """
        # 一次性将tensor转为numpy（减少多次设备拷贝）
        u_np = meta_u.cpu().numpy().astype(np.int64)
        v_np = meta_v.cpu().numpy().astype(np.int64)

        # 构建查询的复合索引
        query_index = pd.MultiIndex.from_arrays([u_np, v_np], names=['u', 'v'])

        # 核心优化：用pandas的get_indexer进行向量化查找（C实现，比map快100+倍）
        idx = time_series.index.get_indexer(query_index)

        # 初始化结果（指定数据类型，与原逻辑一致）
        t_np = np.full(shape=len(u_np), fill_value=default_time, dtype=np.int32)
        # 仅对找到的索引赋值（避免无效操作）
        mask = idx != -1
        t_np[mask] = time_series.iloc[idx[mask]].values

        # 转回tensor并返回原设备
        t_tensor = torch.tensor(t_np, dtype=torch.int32).to(meta_u.device)

        return t_tensor

    result = []
    # 预构建所有边的时间映射（原逻辑不变，优化了内部实现）
    i2p = build_edge_time_mapping(graph, ('ipc', 'assigned to', 'patent'))
    pt2pt = build_edge_time_mapping(graph, ('patent', 'cites', 'patent'))
    pt2pr = build_edge_time_mapping(graph, ('patent', 'cites', 'paper'))

    # 注意：原代码中args.num_metapath_types是全局变量，需确保外部定义，这里保留原逻辑
    for mtype in range(args.num_metapath_types):
        # 优化：用if-elif代替多个独立if，减少冗余条件判断
        if mtype == 0:
            # 提取节点并计算偏移（原逻辑不变，代码稍作整理）
            ipc1 = metapaths[mtype][:, 0]
            pat1 = metapaths[mtype][:, 1] - num_ipc
            ipc2 = metapaths[mtype][:, 2]

            # 查询时间（调用优化后的batch_query_time）
            time1_batch = batch_query_time(ipc1, pat1, i2p)
            time2_batch = batch_query_time(ipc2, pat1, i2p)

            # 拼接结果（原逻辑不变）
            time_result_batch = torch.stack([time1_batch, time2_batch], dim=1)
            result.append(time_result_batch)

        elif mtype == 1:
            ipc1 = metapaths[mtype][:, 0]
            pat1 = metapaths[mtype][:, 1] - num_ipc
            pat2 = metapaths[mtype][:, 2] - num_ipc
            ipc2 = metapaths[mtype][:, 3]

            time1_batch = batch_query_time(ipc1, pat1, i2p)
            time2_batch = batch_query_time(pat1, pat2, pt2pt)
            time3_batch = batch_query_time(ipc2, pat2, i2p)

            time_result_batch = torch.stack([time1_batch, time2_batch, time3_batch], dim=1)
            result.append(time_result_batch)

        elif mtype == 2:
            ipc1 = metapaths[mtype][:, 0]
            pat1 = metapaths[mtype][:, 1] - num_ipc
            par1 = metapaths[mtype][:, 2] - num_ipc - num_patent
            pat2 = metapaths[mtype][:, 3] - num_ipc
            ipc2 = metapaths[mtype][:, 4]

            time1_batch = batch_query_time(ipc1, pat1, i2p)
            time2_batch = batch_query_time(pat1, par1, pt2pr)
            time3_batch = batch_query_time(pat2, par1, pt2pr)
            time4_batch = batch_query_time(ipc2, pat2, i2p)

            time_result_batch = torch.stack([time1_batch, time2_batch, time3_batch, time4_batch], dim=1)
            result.append(time_result_batch)

    return result
def get_evaluation_data(train_adj, test_adj):
    """ Load val/test examples to evaluate link prediction performance"""
    val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(train_adj, test_adj, val_mask_fraction=0.5, test_mask_fraction=0.5)

    return val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(adj, next_adj, val_mask_fraction, test_mask_fraction):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""

    #去除对角线
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    # 所有的测试边
    edges_all = sparse_to_tuple((adj+next_adj)+(adj+next_adj).T)[0]  # All edges in original adj.

    train_edges_all = sparse_to_tuple(adj + adj.T)[0]
    train_edges_all = [tuple(row) for row in train_edges_all]

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    """for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)"""
    for e in edges_next:
        if tuple(e) not in train_edges_all:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))

    #rng = np.random.default_rng(seed=42)
    #rng.shuffle(all_edge_idx)

    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    '''train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])'''

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    # print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return torch.tensor(list(val_edges), dtype=torch.long), torch.tensor(val_edges_false, dtype=torch.long), torch.tensor(list(test_edges), dtype=torch.long), torch.tensor(test_edges_false, dtype=torch.long)

def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def write_to_csv(test_result, output_name, dataset, mod='test'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        print("test result ({})".format(mod), test_result[0], test_result[1])
        # best_auc = test_result
        f.write("{},{},{},{}\n".format(dataset,  mod, "AUC", test_result[0]))
        f.write("{},{},{},{}\n".format(dataset,  mod, "F1", test_result[1]))