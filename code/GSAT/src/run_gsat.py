print('importing...')
import argparse
import os
import os.path as osp
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch

import torch.nn.functional as F
# from drugood.datasets import build_dataset
# from mmcv import Config
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset

import yaml
import scipy
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import logging



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose # 
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected, degree, to_undirected #

from sklearn.metrics import roc_auc_score # 
from rdkit import Chem # 

from pretrain_clf import train_clf_one_seed # 
from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, init_metric_dict
from losses import get_contrast_loss
print('ok!')

class GSAT(nn.Module):

    def __init__(self, clf, clf2, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class, multi_label, random_state,
                 method_config, shared_config, args):
        super().__init__()
        self.clf = clf
        self.clf2 = clf2
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.log_file = model_dir / 'log.txt'
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.fix_sparse_reg = method_config.get('fix_sparse_reg', None)
        self.inc_interval = method_config.get('inc_interval', None)
        self.inc_reg = method_config.get('inc_reg', None)
        self.final_reg = method_config.get('final_reg', None)
        self.init_reg = method_config.get('init_reg', None)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)


        # r, temp, noise, node_attn = args.r, args.temp, args.noise, args.node_attn 
        self.method = args.method
        print(self.method)
        if self.method == 'gsat': # GSAT
            self.__loss__ = self.__loss__1 
            self.concrete_sample = self.concrete_sample1
        elif self.method == 'erm': # GNN
            self.__loss__ = self.__loss__2 
            self.concrete_sample = self.concrete_sample1
        elif self.method == 'gstopr': # ours
            self.__loss__ = self.__loss__2
            self.concrete_sample = self.concrete_sample2
        elif self.method == 'macro': # ours
            self.__loss__ = self.__loss__2
            self.concrete_sample = self.concrete_sample3


        # our configs
        self.args = args
        self.r = args.r
        self.temp = args.temp 
        self.noise = args.noise 
        self.node_attn = args.node_attn 

        self.c_in = args.c_in

        # training
        self.pretrain = args.pretrain
        self.early_stopping = args.early_stopping

    def __loss__1(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        # r: prior
        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def __loss__2(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)
        loss = pred_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        # get edge att
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr) # node embs
        att_log_logits = self.extractor(emb, data.edge_index, data.batch) # edge causal prob logits
        edge_batch = data.batch[data.edge_index[0]] # 

        # graph_reprs = self.clf.get_graph_repr(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        ratio_x = self.r # self.clf.x_out(graph_reprs)  #  #

        att = self.sampling(att_log_logits, epoch, training, edge_batch, r=ratio_x) 

        if self.learn_edge_att:
            if is_undirected(data.edge_index): # undirected?
                trans_idx, trans_val = transpose(data.edge_index, att, None, None, coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else: # this way
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)


        # pred with edge att
        if self.method == 'gsat':
            clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att) 
        elif self.method in ['gstopr', 'macro']: # we use 2 GNNs, one to get edge attention, the other for prediction.
            if self.c_in == 'raw':
                clf_logits = self.clf2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att) 
            elif self.c_in == 'feat':
                clf_logits = self.clf2(emb, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att) 
        elif self.method == 'erm':
            clf_logits = self.clf2(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr)
            
        if clf_logits.isnan().all():
            clf_logits[clf_logits.isnan()] = 0
        clf_logits[clf_logits.isnan()] = clf_logits[~clf_logits.isnan()].mean()
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch) # pred loss, info loss

        return edge_att, loss, loss_dict, clf_logits # logits for pred # TODO: graph reprs POOL(emb)

    
    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()
        self.clf2.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu() # att, loss, pred

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()
        self.clf2.train()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        if not loss.isnan():
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.clf.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.clf2.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.extractor.parameters(), 1)
            self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)
        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))
        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch) # forward pass result
            exp_labels = data.edge_label.data.cpu() # explanation
              
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch, data.edge_index)
            desc, _, _, _, _, _ = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                 data.y.data.cpu(), clf_logits, batch=True)
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels), all_att.append(att), all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu()), all_clf_logits.append(clf_logits)

            if idx == loader_len - 1: # end of an epoch, report all info
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att),
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len
                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict, all_exp_labels, all_att,
                                                                                        all_precision_at_k, all_clf_labels, all_clf_logits, batch=False)
            pbar.set_description(desc)

        return att_auroc, precision, clf_acc, clf_roc, avg_loss

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)

        # early stopping 
        cnt = 0
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('gsat_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 5
            main_metric_idx = 3 if 'ogb' in self.dataset_name else 2  # clf_roc or clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
            # if (r == self.final_r or self.fix_r) and epoch > 10 and ((valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
            #                                                          or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
            #                                                              and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):
            if (valid_res[main_metric_idx] >= metric_dict['metric/best_clf_valid']): # >=

                metric_dict = {'metric/best_clf_epoch': epoch, 'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx], 'metric/best_clf_valid': valid_res[main_metric_idx], 'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0], 'metric/best_x_roc_valid': valid_res[0], 'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1], 'metric/best_x_precision_valid': valid_res[1], 'metric/best_x_precision_test': test_res[1]}
                # save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
                # save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))
                cnt = 0
            else:
                cnt += (epoch >= self.pretrain)

            for metric, value in metric_dict.items(): # valid clf acc is mono increasing
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'gsat_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}, Best Test X Pre@5: {metric_dict["metric/best_x_precision_test"]:.3f}')
            print('====================================')
            print('====================================')
            with open(self.log_file, 'a') as f:
                f.write(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}, Best Test X Pre@5: {metric_dict["metric/best_x_precision_test"]:.3f}\n')
            
            if cnt >= self.early_stopping and epoch >= self.pretrain: 
                break 

        save_checkpoint(self.clf, self.model_dir, model_name='gsat_clf_epoch_' + str(epoch))
        save_checkpoint(self.clf2, self.model_dir, model_name='gsat_clf2_epoch_' + str(epoch))
        save_checkpoint(self.extractor, self.model_dir, model_name='gsat_att_epoch_' + str(epoch))

        
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")[9::]
        
        desc = f'[Seed {self.random_state}, Epoch: {epoch}, Time: {datetime_now}]: gsat_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}, Time: {datetime_now}]: gsat_{phase} finished, '
        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'gsat_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch)
        desc += eval_desc
        
        if not batch:
            with open(self.log_file, 'a') as f:
                f.write(desc + '\n')
        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, batch):
        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if 'ogb' in self.dataset_name:
            evaluator = Evaluator(name='-'.join(self.dataset_name.split('_')))
            if clf_logits.isnan().all():
                clf_logits[clf_logits.isnan()] = 0
            clf_logits[clf_logits.isnan()] = clf_logits[~clf_logits.isnan()].mean()
            clf_logits[clf_logits >= 1] == 1 - 1e-5
            clf_roc = evaluator.eval({'y_pred': clf_logits, 'y_true': clf_labels})['rocauc']

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        # print(bkg_att_weights.mean().item(), signal_att_weights.mean().item())
        # with open(self.log_file, 'a') as f:
        #     f.write(f'{bkg_att_weights.mean().item()} {signal_att_weights.mean().item()}\n')
        self.writer.add_histogram(f'gsat_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'gsat_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'gsat_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'gsat_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/gsat_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max()+1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k) # precision of the first k predicted explanation edges
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name,  node_attn=self.node_attn, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')
        # torch.save(imgs, self.model_dir / f'imgs-{tag}-{epoch}.pt')
        

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def get_sparse_reg(self, inc_interval, inc_reg, current_epoch, init_reg=0.02, final_reg=0.1):
        reg = init_reg + current_epoch // inc_interval * inc_reg
        if reg > final_reg:
            reg = final_reg
        return reg

    def sampling(self, att_log_logits, epoch, training, edge_batch, r=None):
        att = self.concrete_sample(att_log_logits, temp=self.temp, training=training, edge_batch=edge_batch, r=r)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    def concrete_sample3(self, att_log_logit, temp, training, edge_batch, r=None, noise=0., max_iter=10, eps=1e-20):
        """
        gumbel hard sinkhorn topr (Macro Version)
        att_log_logit: (e, 1)
        edge_batch: (e, 1), val: 0 ~ BATCH_SIZE - 1, int
        """
        def sample_gumbel(shape, eps=1e-20):
            u = torch.rand(shape)
            return -torch.log(-torch.log(u + eps) + eps)

        temp = self.temp

        atts = (att_log_logit - att_log_logit.mean()) / att_log_logit.std() # norm
        g = sample_gumbel(atts.shape).to(atts.device) 

        d = atts + g * self.noise if training else atts # (e, 1)
        s_min = atts.min() #if degrees[i] > 0 else g.min()
        s_max = atts.max() #if degrees[i] > 0 else g.max()
        D = torch.cat([d - s_min, s_max - d], dim=-1) # (b, e, 2)
        logT = -D / temp
        row_sum = torch.tensor([[logT.shape[0] * (1-r), logT.shape[0] * r]], device=atts.device) # (1, 2)
        for i in range(max_iter):
            logT = logT - torch.logsumexp(logT, dim=-1, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=-2, keepdim=True) # row norm
            logT = logT + torch.log(row_sum + eps)
        T = logT.exp()
        # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1

        return T[:, [1]]


    # @staticmethod
    def concrete_sample2(self, att_log_logit, temp, training, edge_batch, r=None, noise=0., max_iter=10, eps=1e-6):
        """
        gumbel hard sinkhorn topr
        att_log_logit: (e, 1)
        edge_batch: (e, 1), val: 0 ~ BATCH_SIZE - 1, int
        """
        def sample_gumbel(shape, eps=1e-20):
            u = torch.rand(shape)
            return -torch.log(-torch.log(u + eps) + eps)

        # r = self.r # self.clf()
        temp = self.temp

        batch_size = edge_batch.max().long().item() + 1
        degrees = degree(edge_batch).long() # (b,)
        max_len = degrees.max().item()
        atts = torch.zeros(batch_size, max_len, 1, device=att_log_logit.device) # (b, s, 1)
        g = sample_gumbel(atts.shape).to(atts.device) 
        for i in range(batch_size):
            atti = att_log_logit[edge_batch == i] # (ei, 1)
            atti = (atti - atti.mean()) / atti.std() # norm
            atts[i, :degrees[i]] = atti
            atts[i, degrees[i]:] = atti.min() if atti.numel() else g.min()  # sequence padding
            g[i, degrees[i]:] = g[i].min() # sequence padding

        d = atts + g * self.noise if training else atts # (b, e, 1)
        # s_min = atts.min(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.min()
        # s_max = atts.max(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.max()
        # print(s_min.mean(), s_max.mean())
        # D = torch.cat([d - s_min, s_max - d], dim=-1) # (b, e, 2)
        D = torch.cat([d, 1 - d], dim=-1) # (b, e, 2)
        logT = -D / temp
        degrees = degrees.unsqueeze(-1) # (b, 1)
        row_sum = torch.stack((logT.shape[1] - r * degrees + eps, r * degrees + eps), dim=-1) # (b, 1, 2)
        for i in range(max_iter):
            logT = logT - torch.logsumexp(logT, dim=2, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=1, keepdim=True) # row norm
            logT = logT + torch.log(row_sum)
        T = logT.exp()
        # T = (T - T.min() + eps) / (T.max() - T.min() + eps) # value range 0 ~ 1

        Ts = []
        for i in range(batch_size):
            Ti = T[i, :degrees[i], [1]] # T_res
            Ts.append(Ti)
        T = torch.cat(Ts, dim=0)
        return T 

    # @staticmethod
    def concrete_sample1(self, att_log_logit, temp, training, edge_batch, r=None):
        temp = self.temp
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10) # u
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise) # log(u / (1 - u))
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid() # binary concrete
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index # row, col ?
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col]) # batch of edge
        else:
            att_log_logits = self.feature_extractor(emb, batch) # batch of node
        return att_log_logits


def train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, args):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, data_config.get('mutag_x', False))

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, args.node_attn, 'raw')
    
    emb_dim = model_config['hidden_size']
    if args.c_in == 'feat':
        model2 = get_model(emb_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, args.node_attn, 'feat')
    else:
        model2 = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device, args.node_attn, 'raw')
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    if not method_config['from_scratch']:
        print('[INFO] Pretraining the model...')
        train_clf_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, device, random_state,
                           model=model, loaders=loaders, num_class=num_class, aux_info=aux_info)
        pretrain_epochs = local_config['model_config']['pretrain_epochs'] - 1
        load_checkpoint(model, model_dir=log_dir, model_name=f'epoch_{pretrain_epochs}')
    else:
        print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()) + list(model2.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config} # configs
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}

    metric_dict = deepcopy(init_metric_dict) # init, only has keys, values are all 0s
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GSAT...')
    gsat = GSAT(model, model2, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class, aux_info['multi_label'], random_state, method_config, shared_config, args)
    metric_dict = gsat.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    print("!!!", metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GSAT')
    parser.add_argument('--dataset', default='spmotif_0.5', type=str, help='dataset used')
    parser.add_argument('--backbone', default='PNA', type=str, help='backbone model used')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device id, -1 for cpu')

    parser.add_argument('--method', default='gstopr', type=str, help='gsat or gstopr')
    parser.add_argument('--c_in', default='raw', type=str, help='classifier input')

    parser.add_argument('--alpha', default=0, type=float, help='CIGAv1: contrast loss balance')
    parser.add_argument('--lag', default=0, type=float, help='CIGAv2: lagrange loss constraint < 1')
    parser.add_argument('--beta', default=0, type=float, help='sparse loss balance')

    parser.add_argument('--noise', default=1.0, type=float, help='gumbel noise')
    parser.add_argument('--r', default=0.2, type=float, help='causal ratio')
    parser.add_argument('--temp', default=1.0, type=float, help='sinkhorn temperature')
    parser.add_argument('--node_attn', default='max', type=str, help='mean, max, none')

    parser.add_argument('--pretrain', default=0, type=int)
    parser.add_argument('--early_stopping', default=10, type=int)
    args = parser.parse_args()
    dataset_name = args.dataset
    model_name = args.backbone
    cuda_id = args.cuda
    
    # our configs
    r, temp, noise, node_attn = args.r, args.temp, args.noise, args.node_attn 

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    method_name = "GSAT" #args.method

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r')) # dict

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs-01-100epo' / (model_name + '-seed' + str(random_state) + '-' + args.method +'-'+ args.c_in + '-r' + str(r) + '-t' + str(temp) + '-g' + str(int(noise)) + '-' + str(node_attn)  + '-' + time) 
        hparam_dict, metric_dict = train_gsat_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state, args)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs-01-100epo' / ('stat-' + model_name + '-' + args.method +'-'+ args.c_in+ '-r' + str(r) + '-t' + str(temp) + '-g' + str(int(noise)) + '-' + str(node_attn) + '-' + time)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer, log_dir)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    main()