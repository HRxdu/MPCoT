import os
import dgl
import time
import torch
import torch.nn.functional as F
import logging

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
    output_dir = 'result/LLM/model_output/'

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

    # model_dir is not used in this code for saving.

    # utils folder: utils.py, random_walk.py, minibatch.py
    # models folder: layers.py, models.py
    # main folder: train.py
    # eval folder: link_prediction.py

    """
    #1: Train logging format: Create a new log directory for each run (if log_dir is provided as input). 
    Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the 
    contents of the <>.log will get appended per day => one log file per day.

    #2: Model save format: The model is saved inside model_dir. 

    #3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at 
    each 
    time step. By default, a directory named "output" is created.

    #4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written 
    as per each day => new log file for each day.
    """

    # Load graphs and features.
    device = args.device
    graph = dgl.load_graphs(args.graph_path)[0][0]
    graphs = dgl.load_graphs(args.graphs_path)[0]
    feature_dict = graph.ndata['f'] # keys(['ipc', 'paper', 'patent'])
    del graph

    #train_graph = get_train_graph()
    #test_graph = get_test_graph()

    train_graph,test_graph=get_static_train_test_graph()

    """# 获取图中所有边的索引
    edges_g1 = set(zip(train_graph.edges()[0].tolist(), train_graph.edges()[1].tolist()))
    edges_g2 = set(zip(test_graph.edges()[0].tolist(), test_graph.edges()[1].tolist()))

    # 检查是否有相同的边
    same_edges = edges_g1.intersection(edges_g2)

    # 打印结果 有重合边的原因可能是因为建图时没有去重
    if len(same_edges) > 0:
        print("相同的边:")
        for edge in same_edges:
            print(edge)
    else:
        print("两个图没有相同的边")"""


    train_adj = train_graph.adjacency_matrix(scipy_fmt='csr')
    test_adj = test_graph.adjacency_matrix(scipy_fmt='csr')

    dataset = LLMDataset_node(train_graph, args=args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_edges, val_edges_false, test_edges, test_edges_false = get_evaluation_data(train_adj, test_adj)
    valid_dataset = LLMDataset_eval(val_edges, val_edges_false)
    val_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    test_dataset = LLMDataset_eval(test_edges, test_edges_false)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # model
    model = MTHG(feature_dict, args).to(device)
    opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    early_stop = EarlyStopping(patience=args.patience, verbose=True, save_path=MODEL_DIR + '/checkpoint.pt')
    epochs_val_result = []
    state_dicts = []

    nodes = train_graph.nodes(ntype='ipc')
    # metapaths = get_metapaths_till_t(nodes, timestamp=args.max_timestamp)
    # metapath_graphs = get_metapath_graphs_till_t(metapaths, timestamp=args.max_timestamp)

    for epoch in range(args.num_epochs):

        print('Epoch: %04d' % (epoch + 1))
        # training loop
        model.train()
        mode = 'train'
        epoch_time = 0.0
        epoch_loss = 0.0
        it = 0
        for batch in dataloader:
            t_train = time.time()
            batch_loss = []
            # 传入每个batch，获取batch中所有的ipc节点all_nodes，以及这些节点在每年对应的正负边[共现边]edges_dict
            all_nodes, edges_dict = get_node_pairs([train_graph,test_graph], batch)
            # 将batch中所有的ipc节点转换为索引id
            nodes2id = dict(zip(all_nodes.cpu().numpy().tolist(), list(range(all_nodes.shape[0]))))
            # 取出2010-2022年三种元路径，但是每个每个年份下都是三种元路径列表，其中，每种元路径列表存储了各ipc节点对应年的元路径
            metapaths = get_metapaths(all_nodes, timestamp=args.max_timestamp)
            metapath_graphs = get_metapath_graphs(metapaths, timestamp=args.max_timestamp)
            # 将该批次的nodes及相关数据传入模型，计算特征嵌入
            embd = model([all_nodes, metapaths, metapath_graphs, args.max_timestamp, mode])

            # for tidx, t in enumerate(args.time_range):
            pu, pv = edges_dict['pos']
            nu, nv = edges_dict['neg']

            pu = list(map(lambda x: nodes2id[x], pu.cpu().numpy().tolist()))
            pv = list(map(lambda x: nodes2id[x], pv.cpu().numpy().tolist()))
            nu = list(map(lambda x: nodes2id[x], nu.cpu().numpy().tolist()))
            nv = list(map(lambda x: nodes2id[x], nv.cpu().numpy().tolist()))

            # 计算正负边batch节点在t年的嵌入
            embd_pu = embd[pu]
            embd_pv = embd[pv]
            embd_nu = embd[nu]
            embd_nv = embd[nv]

            embd_pu = embd_pu.view(-1, 1, embd_pu.shape[-1])
            embd_pv = embd_pv.view(-1, embd_pv.shape[-1], 1)
            embd_nu = embd_nu.view(-1, 1, embd_nu.shape[-1])
            embd_nv = embd_nv.view(-1, embd_nv.shape[-1], 1)

            # 计算batch节点正负边在t年的内积(有连边的概率)
            pos_out = torch.bmm(embd_pu, embd_pv).flatten()
            neg_out = -torch.bmm(embd_nu, embd_nv).flatten()

            # 计算batch节点正负边在t年的损失，添加到batch_loss中
            loss_t = -torch.sum(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

            batch_loss.append(torch.unsqueeze(loss_t, dim=0))

            # 将批次中的所有损失合并。
            batch_loss = torch.cat(batch_loss, dim=0)
            # 对合并后的损失求和。
            batch_loss = torch.sum(batch_loss, dim=0)
            epoch_time += time.time() - t_train

            print("Mini batch Iter: {} train_loss= {:.5f}".format(it, batch_loss))
            logging.info("Mini batch Iter: {} train_loss= {:.5f}".format(it, batch_loss))

            # autograd
            # 清除之前的梯度。反向传播损失。更新模型参数。
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            # 更新训练轮的总损失。
            epoch_loss += batch_loss
            # batch次数
            it += 1

        # 该轮次的平均batch损失
        epoch_loss /= it
        print("Time for epoch ", epoch_time)
        logging.info("Time for epoch : {}".format(epoch_time))
        print("Mean Loss at epoch {} : {}".format(epoch, epoch_loss))

        # validation
        # 每一轮所有batch更新完后进行验证
        model.eval()
        mode = 'eval'
        # 初始化两个列表来收集验证集的预测结果。
        val_pred_pos = []
        val_pred_neg = []

        epoch_loss_val = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_loss_val = []
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

                val_pred_pos.append(torch.sigmoid(pos_out))
                val_pred_neg.append(torch.sigmoid(neg_out))

                # 计算loss
                loss_val = -torch.sum(F.logsigmoid(pos_out) + F.logsigmoid(-neg_out))
                batch_loss_val.append(torch.unsqueeze(loss_val, dim=0))

                # 将批次中的所有损失合并。
                batch_loss_val = torch.cat(batch_loss_val, dim=0)
                # 对合并后的损失求和。
                batch_loss_val = torch.sum(batch_loss_val, dim=0)
                epoch_loss_val += batch_loss_val

        # 合并正负样本的预测结果。
        val_pred = torch.cat(val_pred_pos + val_pred_neg).cpu().numpy()
        # 定义验证集的真实标签。
        val_true = np.array([1] * (val_pred.shape[0] // 2) + [0] * (val_pred.shape[0] // 2))
        # 计算auc
        auc_val = roc_auc_score(val_true, val_pred)
        ap_val = average_precision_score(val_true, val_pred)
        f1_val = f1_score(val_true, [1 if i >= 0.5 else 0 for i in val_pred])
        torch.save(model.state_dict(), f"result/LLM/model_output/model/checkpoint_ep{epoch + 1}.pt")

        # 将当前轮的验证AUC得分添加到结果列表。
        epochs_val_result.append(auc_val)
        print("Epoch {}, Val AUC {},Val AP {},Val F1 {}".format(epoch, auc_val, ap_val, f1_val))
        logging.info("Val results at epoch {}: AUC: {}".format(epoch, auc_val))
        # 使用了负的AUC值作为性能指标，那么当AUC值的绝对值开始增加（即实际AUC值开始减小）时，也可能触发早停。
        early_stop(-auc_val, model)
        if early_stop.early_stop:
            print('Early stopping!')
            # Choose best model by validation set performance.
            best_epoch = epochs_val_result.index(max(epochs_val_result))
            print("Best epoch ", best_epoch)
            logging.info("Best epoch {}".format(best_epoch))
            break

        """epoch_loss_val /= it
        # 将当前轮的验证AUC得分添加到结果列表。
        epochs_val_result.append(epoch_loss_val)
        print("Epoch {}, Val loss {}".format(epoch, epoch_loss_val))
        print("Epoch {}, Val AUC {}".format(epoch, auc_val))
        logging.info("Val results at epoch {}: loss: {}".format(epoch, epoch_loss_val))
        # 使用了负的AUC值作为性能指标，那么当AUC值的绝对值开始增加（即实际AUC值开始减小）时，也可能触发早停。
        early_stop(epoch_loss_val, model)
        if early_stop.early_stop:
            print('Early stopping!')
            # Choose best model by validation set performance.
            best_epoch = epochs_val_result.index(min(epochs_val_result))
            print("Best epoch ", best_epoch)
            logging.info("Best epoch {}".format(best_epoch))
            break"""

    # test
    # 加载之前保存的最佳模型状态，然后在测试集上进行预测
    model.load_state_dict(torch.load(MODEL_DIR + '/checkpoint.pt'))
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
    f1_test = f1_score(test_true, [1 if i >= 0.5 else 0 for i in test_pred])

    lk_test_result = {
        "test_pred_pos": test_pred_pos,
        "test_pred_neg": test_pred_neg,
        "test_pred_pos_id": test_pred_pos_id,
        "test_pred_neg_id": test_pred_neg_id
    }
    with open(SAVE_DIR + '/lk_test_result.pkl', 'wb') as file:  # 注意使用二进制写入模式 'wb'
        pickle.dump(lk_test_result, file)

    print("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    logging.info("Best epoch test result auc={} ap={} f1={}\n".format(auc_test, ap_test, f1_test))
    write_to_csv(test_result=[auc_test, ap_test, f1_test], output_name=output_file, dataset=args.dataset)