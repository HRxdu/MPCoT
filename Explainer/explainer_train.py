from Explainer.explainer import *
from Weight_Extraction.weight_extraction import *
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def get_new_edges(test_pred_pos,test_pred_pos_id,threshold):
    test_pred_pos_merged = torch.cat(test_pred_pos).cpu().numpy()
    test_pred_pos_id_merged = [item for sublist in test_pred_pos_id for item in sublist]
    indices = np.where(test_pred_pos_merged > threshold)[0]
    result = [test_pred_pos_id_merged[i] for i in indices]
    return result

def combine_metapath_mask_logit(metapath_mask_logit_list):
    result_df = pd.DataFrame(columns=['mask_logit','metapath'])

    for df in metapath_mask_logit_list:
        column_names = df.columns
        new_names = {f'{column_names[0]}': 'mask_logit', f'{column_names[1]}': 'metapath'}
        df.rename(columns=new_names, inplace=True)

    for df in metapath_mask_logit_list:
        result_df = pd.concat([result_df, df], axis=0, ignore_index=True)

    return result_df

if __name__ == '__main__':
    toa_model_dir = args.toa_model_dir

    # TOA模型存储路径
    data_dir = 'result/{}/'.format(args.dataset)
    output_dir = data_dir + 'model_output/'
    model_dir = output_dir + 'model/checkpoint.pt'
    model_path = toa_model_dir + '/' + model_dir

    # 导入测试集结果
    with open(toa_model_dir + '/result/{}/model_output/output/lk_test_result.pkl'.format(args.dataset), 'rb') as file:
        # 加载文件内容
        lk_test_result = pickle.load(file)

    threshold=args.threshold
    test_pred = torch.cat(lk_test_result['test_pred_pos'] + lk_test_result['test_pred_neg']).cpu().numpy()
    test_true = np.array([1] * (test_pred.shape[0] // 2) + [0] * (test_pred.shape[0] // 2))
    auc_test = roc_auc_score(test_true, test_pred)
    ap_test = average_precision_score(test_true, test_pred)
    f1_test = f1_score(test_true, [1 if i >= threshold else 0 for i in test_pred])

    print('阈值为{}时:auc={} ap={} f1={}\n'.format(threshold,auc_test, ap_test, f1_test))

    test_pred_pos=lk_test_result['test_pred_pos']
    test_pred_pos_id=lk_test_result['test_pred_pos_id']

    new_edges = get_new_edges(test_pred_pos,test_pred_pos_id,threshold)
    with open('output/new_edges_test_TP.pkl', 'wb') as f:
        pickle.dump(new_edges, f)

    for idx,e in enumerate(new_edges):
        trans_e = (int(e[0].item()),int(e[1].item()))
        new_edges[idx] = trans_e

    result_dict = {}
    result_dict['pearsonr_corr_list'] = []
    result_dict['pearsonr_p_list'] = []
    result_dict['Kendall_corr_list'] = []
    result_dict['Kendall_p_list'] = []
    result_dict['Spearman_corr_list'] = []
    result_dict['Spearman_p_list'] = []
    result_dict['ndcg_list'] = []

    result_dict['mask_num_list'] = []

    result_dict['pred_loss_with_metapath_mask_list'] = []
    result_dict['pred_loss_with_weight_mask_list'] = []
    result_dict['pred_loss_with_logit_mask_list'] = []

    result_dict['pos_out_with_metapath_mask_list'] = []
    result_dict['pos_out_with_weight_mask_list'] = []
    result_dict['pos_out_with_logit_mask_list'] = []

    print(f"共{len(new_edges)}条新边...")

    # 设置需要抽取的数量
    num_to_draw = 1000

    # 创建一个单独的随机数生成器，并设置种子
    rng = np.random.default_rng(seed=42)
    # 使用这个随机数生成器来随机抽取索引
    random_indices = rng.choice(range(len(new_edges)), size=num_to_draw, replace=False)

    # for random_idx in range(0,274):
    for random_idx in range(0, num_to_draw):
        edge_index = random_indices[random_idx]
        torch.cuda.empty_cache()
        print(f"random_idx:{random_idx}:")
        print(f"new_edges的第{edge_index}条边:")
        # 取edges_explain
        edges_explain = [new_edges[edge_index]]

        result_name = edge_index
        result_dir = f'new_edges_{result_name}'

        weight_result_path = f'D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/metapath_Weight_Extracted/{result_dir}'
        mask_logit_result_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\mask_logit_result/{result_dir}'
        mask_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\model/{result_dir}'

        # Load graphs and features.
        device = args.device
        graph = dgl.load_graphs(args.graph_path)[0][0]
        feature_dict = graph.ndata['f']  # keys(['ipc', 'paper', 'patent']

        # 训练解释器
        #explainer = Explainer(edges_explain, feature_dict, model_path, args, result_dir)
        #explainer.Explain_edges()
        #del explainer

        "------------------------------------------------------------------------------"

        # 提取edges_explain的元路径权重
        all_nodes = []
        for edge in edges_explain:
            nodes = list(edge)
            all_nodes += nodes
        all_nodes = list(set(all_nodes))
        nodes2id = dict(zip(all_nodes, list(range(len(all_nodes)))))

        Weight_Extract(all_nodes, weight_result_path)

        # 读取edges_explain的权重和掩码logit
        with open(f'{weight_result_path}/weight_extracted.pkl', 'rb') as file:
            # 加载testipc抽出的权重，为数据框列表，每个数据框是各类型元路径的权重
            weight_list = pickle.load(file)

        with open(f'{mask_logit_result_dir}/metapath_mask_logit_result.pkl', 'rb') as file:
            metapath_mask_logit_list = pickle.load(file)

        # 将所有元路径抽出的权重和mask logit合并为一个数据框
        weight_metapath = weight_sort(weight_list)
        mask_logit_metapath = combine_metapath_mask_logit(metapath_mask_logit_list)

        weight_metapath['metapath'] = weight_metapath['metapath'].apply(tuple)
        mask_logit_metapath['metapath'] = mask_logit_metapath['metapath'].apply(tuple)

        merged_df = pd.merge(weight_metapath, mask_logit_metapath, on='metapath')

        """
        ------------------------------------------------------------------------------------------------------
        """

        # 提取出mask
        with open(f'{mask_dir}/metapath_masks.pkl', 'rb') as f:
            metapath_masks = pickle.load(f)

        # 提取出mask对应的metapaths
        metapaths = []
        for mtype in range(args.num_metapath_types):
            metapath = torch.Tensor(metapath_mask_logit_list[mtype]['metapath'].tolist()).long().to(device)
            metapaths.append(metapath)

        # 计算出mask有多少条边
        mask_num = 0
        for mtype, metapath_mask in metapath_masks.items():
            metapath_mask = torch.where(metapath_mask > 0.5, torch.ones_like(metapath_mask),
                                        torch.zeros_like(metapath_mask))
            t_num = int(metapath_mask.sum().item())
            mask_num += t_num
        print("mask有多少条边:", mask_num)
        result_dict['mask_num_list'].append(mask_num)

        # 取出权重前mask_num条边
        weight_metapath_top = weight_metapath.head(mask_num)['metapath'].tolist()

        # 构建权重的mask
        weight_masks = []
        for mtype in range(args.num_metapath_types):
            mtype_weight_mask = []
            for index, row in metapath_mask_logit_list[mtype].iterrows():
                tur_metap = tuple(row['metapath'])
                if tur_metap in weight_metapath_top:
                    mtype_weight_mask.append(1)
                else:
                    mtype_weight_mask.append(0)
            mtype_weight_mask = torch.Tensor(mtype_weight_mask).unsqueeze(1).to(device)
            weight_masks.append(mtype_weight_mask)

        # 构建logit的mask
        logit_sorted = merged_df.sort_values(by='mask_logit', ascending=False)
        logit_metapath_top = logit_sorted.head(mask_num)['metapath'].tolist()
        logit_masks = []
        for mtype in range(args.num_metapath_types):
            mtype_logit_mask = []
            for index, row in metapath_mask_logit_list[mtype].iterrows():
                tur_metap = tuple(row['metapath'])
                if tur_metap in logit_metapath_top:
                    mtype_logit_mask.append(1)
                else:
                    mtype_logit_mask.append(0)
            mtype_logit_mask = torch.Tensor(mtype_logit_mask).unsqueeze(1).to(device)
            logit_masks.append(mtype_logit_mask)

        with torch.no_grad():
            # 构建模型
            mode = 'eval'
            model = MTHG(feature_dict, args).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            metapath_graphs = get_metapath_graphs(metapaths, timestamp=args.test_timestamp)
            embd_with_metapath_mask = model(
                [torch.tensor(all_nodes, dtype=torch.long, device=args.device), metapaths, metapath_graphs,
                 args.test_timestamp, mode],
                metapath_masks['t1'], metapath_masks['t2'],
                metapath_masks['t3'])
            embd_with_weight_mask = model(
                [torch.tensor(all_nodes, dtype=torch.long, device=args.device), metapaths, metapath_graphs,
                 args.test_timestamp, mode],
                weight_masks[0], weight_masks[1],
                weight_masks[2])
            embd_with_logit_mask = model(
                [torch.tensor(all_nodes, dtype=torch.long, device=args.device), metapaths, metapath_graphs,
                 args.test_timestamp, mode],
                logit_masks[0], logit_masks[1],
                logit_masks[2])

            pu = [edge[0] for edge in edges_explain]
            pv = [edge[1] for edge in edges_explain]
            pu = list(map(lambda x: nodes2id[x], pu))
            pv = list(map(lambda x: nodes2id[x], pv))

            embd_pu_with_metapath_mask = embd_with_metapath_mask[pu]
            embd_pv_with_metapath_mask = embd_with_metapath_mask[pv]
            embd_pu_with_metapath_mask = embd_pu_with_metapath_mask.view(-1, 1, embd_pu_with_metapath_mask.shape[-1])
            embd_pv_with_metapath_mask = embd_pv_with_metapath_mask.view(-1, embd_pv_with_metapath_mask.shape[-1], 1)

            pos_out_with_metapath_mask = torch.bmm(embd_pu_with_metapath_mask, embd_pv_with_metapath_mask).flatten()
            pos_out_with_metapath_mask = torch.sigmoid(pos_out_with_metapath_mask)

            embd_pu_with_weight_mask = embd_with_weight_mask[pu]
            embd_pv_with_weight_mask = embd_with_weight_mask[pv]
            embd_pu_with_weight_mask = embd_pu_with_weight_mask.view(-1, 1, embd_pu_with_weight_mask.shape[-1])
            embd_pv_with_weight_mask = embd_pv_with_weight_mask.view(-1, embd_pv_with_weight_mask.shape[-1], 1)

            pos_out_with_weight_mask = torch.bmm(embd_pu_with_weight_mask, embd_pv_with_weight_mask).flatten()
            pos_out_with_weight_mask = torch.sigmoid(pos_out_with_weight_mask)

            embd_pu_with_logit_mask = embd_with_logit_mask[pu]
            embd_pv_with_logit_mask = embd_with_logit_mask[pv]
            embd_pu_with_logit_mask = embd_pu_with_logit_mask.view(-1, 1, embd_pu_with_logit_mask.shape[-1])
            embd_pv_with_logit_mask = embd_pv_with_logit_mask.view(-1, embd_pv_with_logit_mask.shape[-1], 1)

            pos_out_with_logit_mask = torch.bmm(embd_pu_with_logit_mask, embd_pv_with_logit_mask).flatten()
            pos_out_with_logit_mask = torch.sigmoid(pos_out_with_logit_mask)

            eps = 1e-5
            pred_loss_with_metapath_mask = -torch.log(pos_out_with_metapath_mask + eps)
            pred_loss_with_metapath_mask = torch.sum(pred_loss_with_metapath_mask)

            pred_loss_with_weight_mask = -torch.log(pos_out_with_weight_mask + eps)
            pred_loss_with_weight_mask = torch.sum(pred_loss_with_weight_mask)

            pred_loss_with_logit_mask = -torch.log(pos_out_with_logit_mask + eps)
            pred_loss_with_logit_mask = torch.sum(pred_loss_with_logit_mask)

            print(f"pred_loss_with_metapath_mask:{pred_loss_with_metapath_mask}")
            print(f"pred_loss_with_weight_mask:{pred_loss_with_weight_mask}")
            print(f"pred_loss_with_logit_mask:{pred_loss_with_logit_mask}")
            result_dict['pred_loss_with_metapath_mask_list'].append(pred_loss_with_metapath_mask)
            result_dict['pred_loss_with_weight_mask_list'].append(pred_loss_with_weight_mask)
            result_dict['pred_loss_with_logit_mask_list'].append(pred_loss_with_logit_mask)

            result_dict['pos_out_with_metapath_mask_list'].append(pos_out_with_metapath_mask)
            result_dict['pos_out_with_weight_mask_list'].append(pos_out_with_weight_mask)
            result_dict['pos_out_with_logit_mask_list'].append(pos_out_with_logit_mask)

            del model
        with open('output/comparison_result.pkl', 'wb') as file:
            pickle.dump(result_dict, file)




