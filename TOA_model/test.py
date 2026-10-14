import os
import dgl
import time
import torch
import torch.nn.functional as F
import logging
import json

from myModel import MTHG
from arguments import args
from datetime import datetime
from utils.pytorchtools import EarlyStopping
from utils.dataset import LLMDataset_node, LLMDataset_eval, LLMDataset_t, LLMDataset_edge
from utils.util import *
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

if __name__ == '__main__':
    # Assumes a saved base model as input and model name to get the right directory.
    output_dir = 'result/{}/model_output/'.format(args.dataset)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Set paths of sub-directories.
    LOG_DIR = output_dir + args.log_dir
    SAVE_DIR = output_dir + args.save_dir
    CSV_DIR = output_dir + args.csv_dir
    MODEL_DIR = output_dir + args.model_dir

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.isdir(CSV_DIR):
        os.mkdir(CSV_DIR)

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    today = datetime.today()

    # Setup logging
    log_file = LOG_DIR + '/%s_%s_%s_%s.log' % (args.dataset, str(today.year),
                                               str(today.month), str(today.day))

    log_level = logging.INFO
    logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    logging.info(args)

    # Create file name for result log csv from certain flag parameters.
    output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (args.dataset, str(today.year),
                                                  str(today.month), str(today.day))

    # Load graphs and features.
    device = args.device
    graph = dgl.load_graphs(args.graph_path)[0][0]
    graphs = dgl.load_graphs(args.graphs_path)[0]
    feature_dict = graph.ndata['f']  # keys(['ipc', 'paper', 'patent'])
    del graph

    # train_graph = get_train_graph()
    # test_graph = get_test_graph()

    train_graph, test_graph = get_static_train_test_graph()

    # model
    model = MTHG(feature_dict, args).to(device)
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience, verbose=True, save_path=MODEL_DIR + '/checkpoint.pt')
    epochs_val_result = []
    state_dicts = []

    nodes = train_graph.nodes(ntype='ipc')

    train_adj = train_graph.adjacency_matrix(scipy_fmt='csr')
    test_adj = test_graph.adjacency_matrix(scipy_fmt='csr')

    dataset = LLMDataset_node(train_graph, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(train_adj, test_adj)
    valid_dataset = LLMDataset_eval(val_edges, val_edges_false)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    test_dataset = LLMDataset_eval(test_edges, test_edges_false)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # test
    data_dir = 'result/{}/'.format(args.dataset)
    id2ipc = json.load(open(data_dir + 'id2ipc.json', 'r'))

    # 加载之前保存的最佳模型状态，然后在测试集上进行预测
    model.load_state_dict(torch.load(r"D:\Pycharm_project\interpretable_TOA\static_mode\TOA_model\result\大模型\model_output\model\checkpoint.pt"))
    model.eval()
    mode = 'eval'
    # 初始化空列表 test_pred_pos 和 test_pred_neg，用于存储正样本和负样本的预测结果。
    test_pred_pos = []
    test_pred_neg = []
    test_pred_pos_id = []
    test_pred_neg_id = []

    with torch.no_grad():
        for batch in test_dataloader:
            pu, pv, nu, nv = batch

            metapaths_till_t_pu = get_metapaths(pu,
                                                timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_pu = get_metapath_graphs(metapaths_till_t_pu,
                                                            timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_pv = get_metapaths(pv,
                                                timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_pv = get_metapath_graphs(metapaths_till_t_pv,
                                                            timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_nu = get_metapaths(nu,
                                                timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_nu = get_metapath_graphs(metapaths_till_t_nu,
                                                            timestamp=args.test_timestamp)  # [t, num_mtypes]

            metapaths_till_t_nv = get_metapaths(nv,
                                                timestamp=args.test_timestamp)  # [t, num_mtypes, num_indices, mtype_len]
            metapath_graphs_till_t_nv = get_metapath_graphs(metapaths_till_t_nv,
                                                            timestamp=args.test_timestamp)  # [t, num_mtypes]

            embd_pu = model([pu, metapaths_till_t_pu, metapath_graphs_till_t_pu, args.test_timestamp, mode])
            embd_pv = model([pv, metapaths_till_t_pv, metapath_graphs_till_t_pv, args.test_timestamp, mode])
            embd_nu = model([nu, metapaths_till_t_nu, metapath_graphs_till_t_nu, args.test_timestamp, mode])
            embd_nv = model([nv, metapaths_till_t_nv, metapath_graphs_till_t_nv, args.test_timestamp, mode])

            embd_pu = embd_pu.view(-1, 1, embd_pu.shape[-1])
            embd_pv = embd_pv.view(-1, embd_pv.shape[-1], 1)
            embd_nu = embd_nu.view(-1, 1, embd_nu.shape[-1])
            embd_nv = embd_nv.view(-1, embd_nv.shape[-1], 1)

            pos_out = torch.bmm(embd_pu, embd_pv).flatten()
            neg_out = torch.bmm(embd_nu, embd_nv).flatten()

            test_pred_pos.append(torch.sigmoid(pos_out))
            test_pred_neg.append(torch.sigmoid(neg_out))

            test_pred_pos_id.append(list(zip(pu, pv)))
            test_pred_neg_id.append(list(zip(nu, nv)))

    test_pred = torch.cat(test_pred_pos + test_pred_neg).cpu().numpy()
    test_true = np.array([1] * (test_pred.shape[0] // 2) + [0] * (test_pred.shape[0] // 2))
    auc_test = roc_auc_score(test_true, test_pred)
    ap_test = average_precision_score(test_true, test_pred)
    f1_test = f1_score(test_true, [1 if i >= 0.8 else 0 for i in test_pred])

    lk_test_result={
        "test_pred_pos":test_pred_pos,
        "test_pred_neg":test_pred_neg,
        "test_pred_pos_id":test_pred_pos_id,
        "test_pred_neg_id":test_pred_neg_id
    }
    with open(SAVE_DIR+'/lk_test_result.pkl', 'wb') as file:  # 注意使用二进制写入模式 'wb'
        pickle.dump(lk_test_result, file)

    print("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    #logging.info("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    #write_to_csv(test_result=[auc_test, ap_test, f1_test], output_name=output_file, dataset=args.dataset)