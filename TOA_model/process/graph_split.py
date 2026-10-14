import dgl

def edge_filter(edges, t):
    return edges.data['t'] != t

if __name__ == '__main__':
    result = []
    #for t in range(2010, 2024):
    g = dgl.load_graphs('../result/大模型/thg1.dgl')[0][0]
    del g.ndata['f']
    result.append(dgl.edge_type_subgraph(g, etypes=['co-occurs with']))
    dgl.save_graphs('../result/大模型/target_graphs.dgl', result)