import os
import pickle
import torch

from Weight_Extraction.weight_extraction import *

def get_folders(path, prefix):
    a = os.listdir(path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.startswith(prefix)]

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

    new_deges_path = 'output/model'
    prefix = 'new_edges_'
    new_deges_name = get_folders(new_deges_path,prefix)
    new_edges_idx = [int(i[len(prefix):]) for i in new_deges_name]

    with open('output/new_edges_test_TP.pkl', 'rb') as f:
        new_edges = pickle.load(f)

    """
    --保真度+acc
    --保真度+prob
    """
    metapath_fidelity_acc_list = []
    weight_fidelity_acc_list = []

    metapath_fidelity_prob_list = []
    weight_fidelity_prob_list = []

    """
    --保真度-acc
    --保真度-prob
    """

    metapath_fidelity_acc_list__ = []
    weight_fidelity_acc_list__ = []

    metapath_fidelity_prob_list__ = []
    weight_fidelity_prob_list__ = []

    for edge_index in new_edges_idx:
        edges_explain = [new_edges[edge_index]]

        result_name = edge_index
        result_dir = f'new_edges_{result_name}'

        weight_result_path = f'D:/Pycharm_project/interpretable_TOA/static_mode/Explainer/output/metapath_Weight_Extracted/{result_dir}'
        mask_logit_result_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\mask_logit_result/{result_dir}'
        mask_dir = f'D:\Pycharm_project\interpretable_TOA\static_mode\Explainer\output\model/{result_dir}'

        # Load graphs and features.
        device = args.device
        threshold = args.threshold
        graph = dgl.load_graphs(args.graph_path)[0][0]
        feature_dict = graph.ndata['f']

        # 提取edges_explain的元路径权重
        all_nodes = []
        for edge in edges_explain:
            nodes = list(edge)
            all_nodes += nodes
        all_nodes = list(set(all_nodes))
        nodes2id = dict(zip(all_nodes, list(range(len(all_nodes)))))

        # 读取edges_explain的权重和掩码logit
        with open(f'{weight_result_path}/weight_extracted.pkl', 'rb') as file:
            # 加载testipc抽出的权重，为数据框列表，每个数据框是各类型元路径的权重
            weight_list = pickle.load(file)

        with open(f'{mask_logit_result_dir}/metapath_mask_logit_result.pkl', 'rb') as file:
            # 加载testipc抽出的权重，为数据框列表，每个数据框是各类型元路径的权重
            metapath_mask_logit_list = pickle.load(file)

        # 将所有元路径抽出的权重和mask logit合并为一个数据框
        weight_metapath = weight_sort(weight_list)
        mask_logit_metapath = combine_metapath_mask_logit(metapath_mask_logit_list)

        weight_metapath['metapath'] = weight_metapath['metapath'].apply(tuple)
        mask_logit_metapath['metapath'] = mask_logit_metapath['metapath'].apply(tuple)

        merged_df = pd.merge(weight_metapath, mask_logit_metapath, on='metapath')

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

        # 构建meatapath mask的反转
        metapath_masks_reserve = {}
        for mtype, metapath_mask in metapath_masks.items():
            metapath_mask = torch.where(metapath_mask > 0.5, torch.ones_like(metapath_mask),
                                        torch.zeros_like(metapath_mask))
            metapath_mask_reverse = 1-metapath_mask
            metapath_masks_reserve[mtype] = metapath_mask_reverse

        # 构建weight mask的反转
        weight_masks_reverse = []
        for mtype in range(args.num_metapath_types):
            wm_reverse = 1-weight_masks[mtype]
            weight_masks_reverse.append(wm_reverse)


        with torch.no_grad():
            # 构建模型
            mode = 'eval'
            model = MTHG(feature_dict, args).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            metapath_graphs = get_metapath_graphs(metapaths, timestamp=args.test_timestamp)
            # 移除解释所涉及的特征后预测结果

            pu = [edge[0] for edge in edges_explain]
            pv = [edge[1] for edge in edges_explain]
            pu = list(map(lambda x: nodes2id[x], pu))
            pv = list(map(lambda x: nodes2id[x], pv))

            # 原始预测
            embd = model(
                [torch.tensor(all_nodes, dtype=torch.long, device=args.device), metapaths, metapath_graphs,
                 args.test_timestamp, mode])
            embd_pu = embd[pu]
            embd_pv = embd[pv]
            embd_pu = embd_pu.view(-1, 1,embd_pu.shape[-1])
            embd_pv = embd_pv.view(-1,embd_pv.shape[-1],1)

            pos_out = torch.bmm(embd_pu,embd_pv).flatten()
            pos_out = torch.sigmoid(pos_out)

            "-------------------计算保真度-----------------"

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

            embd_pu_with_metapath_mask = embd_with_metapath_mask[pu]
            embd_pv_with_metapath_mask = embd_with_metapath_mask[pv]
            embd_pu_with_metapath_mask = embd_pu_with_metapath_mask.view(-1, 1,embd_pu_with_metapath_mask.shape[-1])
            embd_pv_with_metapath_mask = embd_pv_with_metapath_mask.view(-1,embd_pv_with_metapath_mask.shape[-1],1)

            pos_out_with_metapath_mask = torch.bmm(embd_pu_with_metapath_mask,embd_pv_with_metapath_mask).flatten()
            pos_out_with_metapath_mask = torch.sigmoid(pos_out_with_metapath_mask)

            embd_pu_with_weight_mask = embd_with_weight_mask[pu]
            embd_pv_with_weight_mask = embd_with_weight_mask[pv]
            embd_pu_with_weight_mask = embd_pu_with_weight_mask.view(-1, 1, embd_pu_with_weight_mask.shape[-1])
            embd_pv_with_weight_mask = embd_pv_with_weight_mask.view(-1, embd_pv_with_weight_mask.shape[-1], 1)

            pos_out_with_weight_mask = torch.bmm(embd_pu_with_weight_mask, embd_pv_with_weight_mask).flatten()
            pos_out_with_weight_mask = torch.sigmoid(pos_out_with_weight_mask)

            if (pos_out_with_metapath_mask.cpu().numpy().item()>threshold) == (pos_out.cpu().numpy().item()>threshold):
                metapath_fidelity_acc_list__.append(1)
            else:
                metapath_fidelity_acc_list__.append(0)

            if (pos_out_with_weight_mask.cpu().numpy().item()>threshold) == (pos_out.cpu().numpy().item()>threshold):
                weight_fidelity_acc_list__.append(1)
            else:
                weight_fidelity_acc_list__.append(0)

            # 原始预测和去除解释后的差距
            metapath_fidelity_prob_list__.append((pos_out - pos_out_with_metapath_mask).cpu().numpy().item())
            weight_fidelity_prob_list__.append((pos_out - pos_out_with_weight_mask).cpu().numpy().item())

    # 计算acc保真度
    metapath_fidelity_acc = np.mean(1 - np.array(metapath_fidelity_acc_list))
    weight_fidelity_acc = np.mean(1 - np.array(weight_fidelity_acc_list))

    # 计算prob保真度
    metapath_fidelity_prob = np.mean(np.array(metapath_fidelity_prob_list))
    weight_fidelity_prob = np.mean(np.array(weight_fidelity_prob_list))

    #print("metapath的acc保真度+为:{};weight的acc保真度+为:{}".format(metapath_fidelity_acc, weight_fidelity_acc))
    #print("metapath的prob保真度+为:{};weight的prob保真度+为:{}".format(metapath_fidelity_prob, weight_fidelity_prob))

    """
    保真度-
    """
    metapath_fidelity_acc_mean__ = np.mean(1 - np.array(metapath_fidelity_acc_list__))
    weight_fidelity_acc_mean__ = np.mean(1 - np.array(weight_fidelity_acc_list__))

    # 计算prob保真度
    metapath_fidelity_prob_mean__ = np.mean(np.array(metapath_fidelity_prob_list__))
    weight_fidelity_prob_mean__ = np.mean(np.array(weight_fidelity_prob_list__))

    metapath_fidelity_acc_std__ = np.std(1 - np.array(metapath_fidelity_acc_list__))
    weight_fidelity_acc_std__ = np.std(1 - np.array(weight_fidelity_acc_list__))

    # 计算prob保真度
    metapath_fidelity_prob_std__ = np.std(np.array(metapath_fidelity_prob_list__))
    weight_fidelity_prob_std__ = np.std(np.array(weight_fidelity_prob_list__))

    print("metapath的acc保真度-为:\nmean:{} std:{};\nweight的acc保真度-为:\nmean:{} std:{}".format(metapath_fidelity_acc_mean__, metapath_fidelity_acc_std__,weight_fidelity_acc_mean__,weight_fidelity_acc_std__))
    print("metapath的prob保真度-为:\nmean:{} std:{};\nweight的prob保真度-为:\nmean:{} std:{}".format(metapath_fidelity_prob_mean__, metapath_fidelity_prob_std__,weight_fidelity_prob_mean__,weight_fidelity_prob_std__))

