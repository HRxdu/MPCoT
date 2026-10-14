import os
import logging
import time
from datetime import datetime
import pandas as pd

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from myModel import MTHG
from utils.util import *
from pytorchtools import EarlyStopping

class Explainer:
    """
    parameter:
    edges_explain:the ipc turple list you want to explain,Example format is [(977, 1821)]
    model_path:the trained model saved path
    """
    def __init__(
        self,
        edges_explain: list,
        input_feature_dict,
        model_path,
        args,
        result_dir
    ):
        self.edges_explain = edges_explain
        self.input_feature_dict = input_feature_dict
        self.model_path = model_path
        self.args = args
        self.metapath_size = args.metapath_size
        self.feature_size = args.feature_size
        self.result_dir = result_dir
        self.explain_module = Explain_Module(self.edges_explain, self.input_feature_dict, self.args,metapath_size = self.metapath_size,feature_size = self.feature_size).to(args.device)
        self.sample_size = args.sample_size

    def Explain_edges(self):
        output_dir = 'output/'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        LOG_DIR = output_dir + self.args.log_dir
        SAVE_DIR = output_dir + self.args.save_dir
        CSV_DIR = output_dir + self.args.csv_dir
        MODEL_DIR = output_dir + self.args.model_dir

        if not os.path.isdir(LOG_DIR):
            os.mkdir(LOG_DIR)

        if not os.path.isdir(SAVE_DIR):
            os.mkdir(SAVE_DIR)

        if not os.path.isdir(CSV_DIR):
            os.mkdir(CSV_DIR)

        if not os.path.isdir(MODEL_DIR):
            os.mkdir(MODEL_DIR)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.GPU_ID

        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        today = datetime.today()

        # Setup logging
        log_file = LOG_DIR + '/%s_%s_%s_%s.log' % (self.args.dataset, str(today.year),
                                                   str(today.month), str(today.day))

        log_level = logging.INFO
        logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')

        logging.info(self.args)

        # Create file name for result log csv from certain flag parameters.
        output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (args.dataset, str(today.year),
                                                      str(today.month), str(today.day))

        device = self.args.device

        opt = torch.optim.Adam(list(self.explain_module.parameters()), lr=self.args.explain_lr)
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
        early_stop = EarlyStopping(patience=self.args.explain_patience, verbose=True, save_path=MODEL_DIR)

        metapaths = self.explain_module.metapaths
        with open(MODEL_DIR + '/metapaths.pkl', 'wb') as f:
            pickle.dump(metapaths, f)

        epochs_loss_result = []

        for epoch in range(self.args.explain_num_epochs):
            feature_masks={}
            metapath_masks={}
            print('Epoch: %04d' % (epoch + 1))
            self.explain_module.train()
            epoch_time = 0.0
            epoch_loss = 0.0
            t_train = time.time()

            ypred ,feature_masks_samples,metapath_masks_samples= self.explain_module(model_path = self.model_path,sample_num = self.sample_size)
            loss,pred_loss,feature_size_loss,metapath_size_loss,best_sample_idx= self.explain_module.loss(ypred)

            metapath_mask_logit_result = self.get_explaination()

            epoch_time += time.time() - t_train
            epoch_loss = loss.detach().cpu().item()

            print("Time for epoch ", epoch_time)
            logging.info("Time for epoch : {}".format(epoch_time))

            print("Mean Loss at epoch {} : {}".format(epoch + 1, epoch_loss))
            logging.info("Mean Loss at epoch {} : {}".format(epoch + 1, epoch_loss))

            print("pred_loss at epoch {} : {}".format(epoch + 1, pred_loss.detach().cpu().item()))
            logging.info("pred_loss at epoch {} : {}".format(epoch + 1, pred_loss.detach().cpu().item()))

            """print("feature_size_loss at epoch {} : {}".format(epoch + 1, feature_size_loss.detach().cpu().item()))
            logging.info("feature_size_loss at epoch {} : {}".format(epoch + 1, feature_size_loss.detach().cpu().item()))"""

            print("metapath_size_loss at epoch {} : {}".format(epoch + 1, metapath_size_loss.detach().cpu().item()))
            logging.info("metapath_size_loss at epoch {} : {}".format(epoch + 1, metapath_size_loss.detach().cpu().item()))

            epochs_loss_result.append(epoch_loss)

            for node_type, feature_mask in feature_masks_samples.items():
                if feature_masks_samples[node_type] is None:
                    feature_masks[node_type] = None
                else:
                    feature_masks[node_type] = feature_masks_samples[node_type][best_sample_idx]

            for metapath_type, metapath_mask in metapath_masks_samples.items():
                if metapath_masks_samples[metapath_type] is None:
                    metapath_masks[metapath_type] = None
                else:
                    metapath_masks[metapath_type] = metapath_masks_samples[metapath_type][best_sample_idx]

            early_stop(epoch_loss, self.explain_module,feature_masks,metapath_masks,metapath_mask_logit_result,self.result_dir)
            if early_stop.early_stop:
                print('Early stopping!')
                # Choose best model by validation set performance.
                best_epoch = epochs_loss_result.index(min(epochs_loss_result))
                print("Best epoch ", best_epoch)
                logging.info("Best epoch {}".format(best_epoch))
                break

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step(epoch_loss)

    def get_explaination(self):
        metapath_mask_logit_result = self.explain_module.get_metapath_mask_logit()
        return metapath_mask_logit_result

class Explain_Module(nn.Module):
    def __init__(
        self,
        edges_explain:list,
        input_feature_dict,
        args,
        metapath_size,
        feature_size,
        feat_m = False,
        meta_m = True
    ):
        super(Explain_Module, self).__init__()
        self.args = args
        self.edges_explain = edges_explain

        all_nodes = list(set([node for edge in edges_explain for node in edge]))
        all_nodes = torch.tensor(all_nodes, dtype=torch.long, device=args.device)

        self.all_nodes = all_nodes
        self.nodes2id = dict(zip(self.all_nodes.cpu().numpy().tolist(), list(range(self.all_nodes.shape[0]))))

        self.metapaths = get_metapaths(self.all_nodes, timestamp=self.args.test_timestamp)
        self.metapath_graphs = get_metapath_graphs(self.metapaths, timestamp=self.args.test_timestamp)

        self.input_feature_dict = input_feature_dict

        self.feat_m = feat_m
        self.meta_m = meta_m

        if self.feat_m:
            self.ipc_feature_mask_logit = nn.Parameter(
                torch.empty(size=(input_feature_dict['ipc'].shape[1],2)).to(self.args.device))
            self.paper_feature_mask_logit = nn.Parameter(
                torch.empty(size=(input_feature_dict['paper'].shape[1],2)).to(self.args.device))
            self.patent_feature_mask_logit = nn.Parameter(
                torch.empty(size=(input_feature_dict['patent'].shape[1],2)).to(self.args.device))
            with torch.no_grad():
                nn.init.xavier_normal_(self.ipc_feature_mask_logit, gain=1.414)
                nn.init.xavier_normal_(self.paper_feature_mask_logit, gain=1.414)
                nn.init.xavier_normal_(self.patent_feature_mask_logit, gain=1.414)

        if self.meta_m:
            """self.metapath_mask_t1_logit = nn.Parameter(
                torch.empty(size=(self.metapaths[0].shape[0], 2)).to(self.args.device))
            self.metapath_mask_t2_logit = nn.Parameter(
                torch.empty(size=(self.metapaths[1].shape[0], 2)).to(self.args.device))
            self.metapath_mask_t3_logit = nn.Parameter(
                torch.empty(size=(self.metapaths[2].shape[0], 2)).to(self.args.device))
            with torch.no_grad():
                nn.init.xavier_normal_(self.metapath_mask_t1_logit, gain=1.414)
                nn.init.xavier_normal_(self.metapath_mask_t2_logit, gain=1.414)
                nn.init.xavier_normal_(self.metapath_mask_t3_logit, gain=1.414)"""

            self.metapath_mask_t1_logit = torch.zeros(size=(self.metapaths[0].shape[0], 2)).to(self.args.device)
            self.metapath_mask_t1_logit = nn.Parameter(self.metapath_mask_t1_logit)

            self.metapath_mask_t2_logit = torch.zeros(size=(self.metapaths[1].shape[0], 2)).to(self.args.device)
            self.metapath_mask_t2_logit = nn.Parameter(self.metapath_mask_t2_logit)

            self.metapath_mask_t3_logit = torch.zeros(size=(self.metapaths[2].shape[0], 2)).to(self.args.device)
            self.metapath_mask_t3_logit = nn.Parameter(self.metapath_mask_t3_logit)

        self.coeffs = {
            "metapath_size": metapath_size,
            "feature_size": feature_size,
        }

    def construct_feature_masks(self,feature_mask_logit):
        feature_mask = self.gumbel_softmax_sampling(feature_mask_logit)[:, 1]
        feature_mask = feature_mask.unsqueeze(0)
        return feature_mask

    def construct_metapath_masks(self,metapath_mask_logit):
        metapath_mask = self.gumbel_softmax_sampling(metapath_mask_logit)[:, 1]
        metapath_mask = metapath_mask.unsqueeze(1)
        return metapath_mask

    def get_metapath_mask_logit(self):
        df_list = []
        for mtype in range(self.args.num_metapath_types):
            if mtype == 0:
                metapath_type = self.metapaths[mtype].cpu().numpy().tolist()
                mask_logit_type = torch.softmax(self.metapath_mask_t1_logit,dim=1)[:, 1].cpu().flatten().tolist()
                df = pd.DataFrame({
                    f'mask_logit_mtype{mtype}': mask_logit_type,
                    f'metapath_mtype{mtype}': metapath_type
                })

            if mtype == 1:
                metapath_type = self.metapaths[mtype].cpu().numpy().tolist()
                mask_logit_type = torch.softmax(self.metapath_mask_t2_logit,dim=1)[:, 1].cpu().flatten().tolist()
                df = pd.DataFrame({
                    f'mask_logit_mtype{mtype}': mask_logit_type,
                    f'metapath_mtype{mtype}': metapath_type
                })

            if mtype == 2:
                metapath_type = self.metapaths[mtype].cpu().numpy().tolist()
                mask_logit_type = torch.softmax(self.metapath_mask_t3_logit,dim=1)[:, 1].cpu().flatten().tolist()
                df = pd.DataFrame({
                    f'mask_logit_mtype{mtype}': mask_logit_type,
                    f'metapath_mtype{mtype}': metapath_type
                })

            df_list.append(df)
        return df_list

    def inverse_gumbel_cdf(self,y, mu, beta):
        return mu - beta * torch.log(-torch.log(y))

    def gumbel_softmax_sampling(self,h, mu=0, beta=1, tau=0.01):
        """
        h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
        """
        shape_h = h.shape
        p = F.softmax(h, dim=1)
        y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
        g = self.inverse_gumbel_cdf(y, mu, beta).to(self.args.device)
        x = torch.log(p) + g  # samples follow Gumbel distribution.
        # using softmax to generate one_hot vector:
        x = x / tau
        x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
        return x

    def get_ipc_feature_mask(self):
        ipc_feature_mask_logit = torch.empty(size=(self.input_feature_dict['ipc'].shape[1], 2)).to(self.args.device)
        nn.init.xavier_normal_(ipc_feature_mask_logit, gain=1.414)
        ipc_feature_mask = self.gumbel_softmax_sampling(ipc_feature_mask_logit)[:, 1]
        ipc_feature_mask = ipc_feature_mask.unsqueeze(0)
        return ipc_feature_mask

    def forward(self,model_path,sample_num = 1):
        device = self.args.device
        mode = 'eval'

        if self.feat_m:
            self.ipc_feature_mask_samples = []
            self.paper_feature_mask_samples = []
            self.patent_feature_mask_samples = []
        else:
            self.ipc_feature_mask_samples = None
            self.paper_feature_mask_samples = None
            self.patent_feature_mask_samples = None

        if self.meta_m:
            self.metapath_mask_t1_samples = []
            self.metapath_mask_t2_samples = []
            self.metapath_mask_t3_samples = []
        else:
            self.metapath_mask_t1_samples = None
            self.metapath_mask_t2_samples = None
            self.metapath_mask_t3_samples = None

        pos_out_samples = []

        for num in range(sample_num):
            if self.feat_m:
                ipc_feature_mask = self.construct_feature_masks(self.ipc_feature_mask_logit)
                paper_feature_mask = self.construct_feature_masks(self.paper_feature_mask_logit)
                patent_feature_mask = self.construct_feature_masks(self.patent_feature_mask_logit)

                self.ipc_feature_mask_samples.append(ipc_feature_mask)
                self.paper_feature_mask_samples.append(paper_feature_mask)
                self.patent_feature_mask_samples.append(patent_feature_mask)

            if self.meta_m:
                metapath_mask_t1 = self.construct_metapath_masks(self.metapath_mask_t1_logit)
                metapath_mask_t2 = self.construct_metapath_masks(self.metapath_mask_t2_logit)
                metapath_mask_t3 = self.construct_metapath_masks(self.metapath_mask_t3_logit)

                self.metapath_mask_t1_samples.append(metapath_mask_t1)
                self.metapath_mask_t2_samples.append(metapath_mask_t2)
                self.metapath_mask_t3_samples.append(metapath_mask_t3)

        for num in range(sample_num):
            feature_dict = {}

            if self.feat_m is False:
                feature_dict['ipc'] = self.input_feature_dict['ipc'].to(self.args.device)
                feature_dict['paper'] = self.input_feature_dict['paper'].to(self.args.device)
                feature_dict['patent'] = self.input_feature_dict['patent'].to(self.args.device)
            else:
                feature_dict['ipc'] = self.input_feature_dict['ipc'].to(self.args.device) * \
                                      self.ipc_feature_mask_samples[num]
                feature_dict['paper'] = self.input_feature_dict['paper'].to(self.args.device) * \
                                        self.paper_feature_mask_samples[num]
                feature_dict['patent'] = self.input_feature_dict['patent'].to(self.args.device) * \
                                         self.patent_feature_mask_samples[num]

            model = MTHG(feature_dict, self.args).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            if self.meta_m:
                embd = model([self.all_nodes, self.metapaths, self.metapath_graphs, args.test_timestamp, mode],
                             self.metapath_mask_t1_samples[num],self.metapath_mask_t2_samples[num],self.metapath_mask_t3_samples[num])
            else:
                embd = model([self.all_nodes, self.metapaths, self.metapath_graphs, args.test_timestamp, mode])

            pu = [edge[0] for edge in self.edges_explain]
            pv = [edge[1] for edge in self.edges_explain]
            pu = list(map(lambda x: self.nodes2id[x], pu))
            pv = list(map(lambda x: self.nodes2id[x], pv))

            embd_pu = embd[pu]
            embd_pv = embd[pv]
            embd_pu = embd_pu.view(-1, 1, embd_pu.shape[-1])
            embd_pv = embd_pv.view(-1, embd_pv.shape[-1], 1)

            pos_out = torch.bmm(embd_pu, embd_pv).flatten()
            pos_out = torch.sigmoid(pos_out)

            pos_out_samples.append(pos_out)

        feature_masks_samples = {'ipc':self.ipc_feature_mask_samples,'paper':self.paper_feature_mask_samples,'patent':self.patent_feature_mask_samples}
        metapath_masks_samples = {'t1':self.metapath_mask_t1_samples,'t2':self.metapath_mask_t2_samples,'t3':self.metapath_mask_t3_samples}

        return pos_out_samples,feature_masks_samples,metapath_masks_samples

    def loss(self,ypred_samples):
        pred_loss_samples=[]
        feature_size_loss_samples = []
        metapath_size_loss_samples = []
        loss_samples = []
        eps = 1e-5

        for index,ypred in enumerate(ypred_samples):
            #pred_loss
            pred_loss = -torch.log(ypred+eps) #预测损失，越接近于1越小 元路径采样全是0时，这就是0了
            pred_loss = torch.sum(pred_loss) #求和每个预测的边的损失
            pred_loss_samples.append(pred_loss)
            feature_size_loss = 0.0
            metapath_size_loss = 0.0

            #size_loss
            if self.feat_m:
                feature_size_loss = self.coeffs["feature_size"] * (torch.sum(self.ipc_feature_mask_samples[index]) + torch.sum(self.paper_feature_mask_samples[index]) + torch.sum(self.patent_feature_mask_samples[index]))
                feature_size_loss_samples.append(feature_size_loss)

            if self.meta_m:
                metapath_size_loss = self.coeffs["metapath_size"] * (torch.sum(self.metapath_mask_t1_samples[index]) + torch.sum(self.metapath_mask_t2_samples[index]) + torch.sum(self.metapath_mask_t3_samples[index]))
                metapath_size_loss_samples.append(metapath_size_loss)

            loss_samples.append((pred_loss+feature_size_loss+metapath_size_loss).detach().cpu().item())

        # 平均每个采样的损失
        pred_loss_result = torch.mean(torch.stack(pred_loss_samples, dim=0))

        if len(feature_size_loss_samples) > 0:
            feature_size_loss_result = torch.mean(torch.stack(feature_size_loss_samples, dim=0))
        else:
            feature_size_loss_result = torch.tensor(0.0)

        if len(metapath_size_loss_samples) > 0:
            metapath_size_loss_result = torch.mean(torch.stack(metapath_size_loss_samples, dim=0))
        else:
            metapath_size_loss_result = torch.tensor(0.0)

        #元路径掩码不为0的索引
        non_zero_indices = []
        for index in range(len(ypred_samples)):
            t1_sample = self.metapath_mask_t1_samples[index]
            t2_sample = self.metapath_mask_t2_samples[index]
            t3_sample = self.metapath_mask_t3_samples[index]
            t1_sample = torch.where(t1_sample > 0.5, torch.ones_like(t1_sample),
                                        torch.zeros_like(t1_sample))
            t2_sample = torch.where(t2_sample > 0.5, torch.ones_like(t2_sample),
                                        torch.zeros_like(t2_sample))
            t3_sample = torch.where(t3_sample > 0.5, torch.ones_like(t3_sample),
                                        torch.zeros_like(t3_sample))
            t1_num = int(t1_sample.sum().item())
            t2_num = int(t2_sample.sum().item())
            t3_num = int(t3_sample.sum().item())
            if t1_num!=0 or t2_num!=0 or t3_num!=0:
                non_zero_indices.append(index)

        min_pred_loss = np.inf
        best_sample_idx = None
        for index in non_zero_indices:
            pl = pred_loss_samples[index]
            if pl < min_pred_loss:
                min_pred_loss = pl
                best_sample_idx = index

        loss = pred_loss_result + feature_size_loss_result + metapath_size_loss_result

        return loss,pred_loss_result,feature_size_loss_result,metapath_size_loss_result,best_sample_idx







