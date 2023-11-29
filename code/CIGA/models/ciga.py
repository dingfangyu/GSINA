import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from torch_geometric.utils import degree
from utils.get_subgraph import relabel, split_batch
# from utils.mask import clear_masks, set_masks
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks


from models.conv import GNN_node, GNN_node_Virtualnode
from models.gnn import GNN, LeGNN


class GNNERM(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNERM, self).__init__()
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batch, return_data="pred"):
        causal_pred, causal_rep = self.classifier(batch, get_rep=True)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")



class GNNPooling(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 pooling='asap',
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean"):
        super(GNNPooling, self).__init__()
        if pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            self.pool = ASAPooling(emb_dim, ratio, dropout=drop_ratio)
            edge_dim = -1
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.pooling = pooling

        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=emb_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=emb_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              edge_dim=edge_dim)

    def forward(self, batched_data, return_data="pred"):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        device = x.device
        h = self.gnn_encoder(batched_data)
        edge_weight = None  #torch.ones(edge_index[0].size()).to(device)
        x, edge_index, causal_edge_weight, batch, perm = self.pool(h, edge_index, edge_weight=edge_weight, batch=batch)
        col, row = batched_data.edge_index
        node_mask = torch.zeros(batched_data.x.size(0)).to(device)
        node_mask[perm] = 1
        edge_mask = node_mask[col] * node_mask[row]
        if self.pooling.lower() == 'asap':
            # Cancel out the edge attribute when using ASAP pooling
            # since (1) ASAP not compatible with edge attr
            #       (2) performance of DrugOOD will not be affected w/o edge attr
            edge_attr = torch.ones(row.size()).to(device)

        # causal_x, causal_edge_index, causal_batch, _ = relabel(x, edge_index, batch)
        causal_x, causal_edge_index, causal_batch = x, edge_index, batch
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)
        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            #Nothing will happen for ERM
            return causal_pred, causal_rep
        else:
            raise Exception("Not support return type")


class CIGA(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,
                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 sigma_len=3):
        super(CIGA, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)

        # self.gnn_encoder.reset_parameters()
        # self.classifier.reset_parameters()

        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)



    def forward(self, batch, return_data="pred", return_spu=False, debug=False):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        causal_edge_index = torch.LongTensor([[], []]).to(device)
        causal_edge_weight = torch.tensor([]).to(device)
        causal_edge_attr = torch.tensor([]).to(device)
        spu_edge_index = torch.LongTensor([[], []]).to(device)
        spu_edge_weight = torch.tensor([]).to(device)
        spu_edge_attr = torch.tensor([]).to(device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(batch)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.ratio * N)

            edge_attr = batch.edge_attr[C:C + N]
            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]
            if debug:
                print(n_reserve)
                print(idx_reserve)
                print(idx_drop)
            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            spu_edge_index = torch.cat([spu_edge_index, edge_index[:, idx_drop]], dim=1)

            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            spu_edge_weight = torch.cat([spu_edge_weight, -1 * single_mask[idx_drop]])

            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            spu_edge_attr = torch.cat([spu_edge_attr, edge_attr[idx_drop]])

        if self.c_in.lower() == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        else:
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)

        # obtain \hat{G_c}
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(self.classifier, causal_edge_weight, causal_graph.edge_index)
        # set_masks(causal_edge_weight, self.classifier)
        # obtain predictions with the classifier based on \hat{G_c}
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        # whether to return the \hat{G_s} for further use
        if return_spu:
            spu_graph = DataBatch.Batch(batch=spu_batch,
                                         edge_index=spu_edge_index,
                                         x=spu_x,
                                         edge_attr=spu_edge_attr)
            # set_masks(spu_edge_weight, self.classifier)
            set_masks(self.classifier, spu_edge_weight, spu_graph.edge_index)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(spu_graph, get_rep=True)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
            clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            causal_h, _, __, ___ = relabel(h, causal_edge_index, batch.batch)
            if self.c_pool.lower() == "add":
                casual_rep_from_feat = global_add_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "max":
                casual_rep_from_feat = global_max_pool(causal_h, batch=causal_batch)
            elif self.c_pool.lower() == "mean":
                casual_rep_from_feat = global_mean_pool(causal_h, batch=causal_batch)
            else:
                raise Exception("Not implemented contrastive feature pooling")

            return causal_pred, casual_rep_from_feat
        else:
            return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight

    def get_dir_loss(self, batched_data, labels, criterion, is_labeled=None, return_data="pred"):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch),\
                pred_edge_weight = self.forward(batched_data,return_data="")
        if is_labeled == None:
            is_labeled = torch.ones(labels.size()).to(labels.device)

        def get_comb_pred(predictor, causal_graph_x, spu_graph_x):
            causal_pred = predictor.graph_pred_linear(causal_graph_x)
            spu_pred = predictor.spu_mlp(spu_graph_x).detach()
            return torch.sigmoid(spu_pred) * causal_pred

        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr)
        set_masks(causal_edge_weight, self.classifier)
        causal_pred, causal_rep = self.classifier(causal_graph, get_rep=True)
        clear_masks(self.classifier)

        spu_graph = DataBatch.Batch(batch=spu_batch, edge_index=spu_edge_index, x=spu_x, edge_attr=spu_edge_attr)
        set_masks(spu_edge_weight, self.classifier)
        spu_pred, spu_rep = self.classifier.get_spu_pred(spu_graph, get_rep=True)
        clear_masks(self.classifier)

        env_loss = torch.tensor([]).to(causal_rep.device)
        for spu in spu_rep:
            rep_out = get_comb_pred(self.classifier, causal_rep, spu)
            env_loss = torch.cat([env_loss, criterion(rep_out[is_labeled], labels[is_labeled]).unsqueeze(0)])

        dir_loss = torch.var(env_loss * spu_rep.size(0)) + env_loss.mean()

        if return_data.lower() == "pred":
            return get_comb_pred(causal_rep, spu_rep)
        elif return_data.lower() == "rep":
            return dir_loss, causal_pred, spu_pred, causal_rep
        else:
            return dir_loss, causal_pred


class GSTOPR(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,

                 noise=1.0,
                 temp=1.0,
                 node_attn='none',
                 max_iter=30,
                 gumbel_samples=5,

                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 sigma_len=3):
        super(GSTOPR, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)

        # self.gnn_encoder.reset_parameters()
        # self.classifier.reset_parameters()

        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)

        # GSTOPR args
        self.noise = noise
        self.temp = temp
        self.node_attn = node_attn
        self.max_iter = max_iter
        self.gumbel_samples = gumbel_samples



    def forward(self, batch, eps=1e-20, return_data="pred", return_spu=False, debug=False):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        
        # get edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep)#.view(-1) # edge_attn, \in R (e, 1)

        # atts = (pred_edge_weight - pred_edge_weight.mean()) / pred_edge_weight.std() # norm
        # def sample_gumbel(shape, eps=1e-20):
        #     u = torch.rand(shape)
        #     return -torch.log(-torch.log(u + eps) + eps)
        # g = sample_gumbel(atts.shape).to(atts.device) 

        # d = atts + g * self.noise if self.training else atts # (e, 1)
        # s_min = atts.min() #if degrees[i] > 0 else g.min()
        # s_max = atts.max() #if degrees[i] > 0 else g.max()
        # D = torch.cat([d -s_min, s_max - d], dim=-1) # (b, e, 2)
        # logT = -D / self.temp
        # row_sum = torch.tensor([[logT.shape[0] * (1-self.ratio), logT.shape[0] * self.ratio]], device=atts.device) # (1, 2)
        # for i in range(self.max_iter):
        #     logT = logT - torch.logsumexp(logT, dim=-1, keepdim=True) # col norm
        #     logT = logT - torch.logsumexp(logT, dim=-2, keepdim=True) # row norm
        #     logT = logT + torch.log(row_sum + eps)
        # T = logT.exp()
        # # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1

        # T = T[:, [1]]



        # sample edge weight
        node_batch = batch.batch # (n,)
        edge_batch = node_batch[batch.edge_index[0]] # (e,)
        batch_size = edge_batch.max().long().item() + 1 # b
        degrees = degree(edge_batch).long() # (b,)
        max_len = degrees.max().item() # s
        atts = torch.zeros(batch_size, max_len, 1, device=device) # (b, s, 1)

        def sample_gumbel(shape, eps=1e-20):
            u = torch.rand(shape)
            return -torch.log(-torch.log(u + eps) + eps)

        g = sample_gumbel(atts.shape).to(device) # gumbel noise
        for i in range(batch_size):
            atti = pred_edge_weight[edge_batch == i] # (ei, 1)
            atti = (atti - atti.mean()) / atti.std() # norm
            atts[i, :degrees[i]] = atti
            atts[i, degrees[i]:] = atti.min() #- atti.std() # sequence padding
            g[i, degrees[i]:] = g.min() # sequence padding

        d = atts + g * self.noise if self.training else atts # (b, e, 1)
        # sinkhorn soft topr
        # D = torch.cat([d, 1 - d], dim=-1) # (b, e, 2)
        # s_min = atts.min(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.min()
        # s_max = atts.max(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.max()
        D = torch.cat([d, 1 - d], dim=-1) # (b, e, 2)
        logT = -D / self.temp
        degrees = degrees.unsqueeze(-1) # (b, 1)
        row_sum = torch.stack((logT.shape[1] - self.ratio * degrees + eps, self.ratio * degrees + eps), dim=-1) # (b, 1, 2)
        for i in range(self.max_iter):
            logT = logT - torch.logsumexp(logT, dim=2, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=1, keepdim=True) # row norm
            logT = logT + torch.log(row_sum)
        T = logT.exp()
        # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1

        Ts = []
        for i in range(batch_size):
            Ti = T[i, :degrees[i], [1]] # T_res
            # Ti = T[i, :degrees[i], 1] # T_res
            Ts.append(Ti)
        T = torch.cat(Ts, dim=0) # (e, 1)

        # obtain predictions with the classifier based on the sampled edge attentions
        # set_masks(self.classifier, T, batch.edge_index, apply_sigmoid=False)
        # set_masks(T, self.classifier, apply_sigmoid=False)
        if self.c_in == 'feat':
            batch.x = h 
        causal_pred, causal_rep = self.classifier(batch, get_rep=True, node_attn=self.node_attn, edge_atten=T)
        # clear_masks(self.classifier)
        # return causal_pred
    
        # whether to return the \hat{G_s} for further use
        if return_spu:
            # set_masks(self.classifier, 1-T, batch.edge_index, apply_sigmoid=False)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(batch, get_rep=True, node_attn=self.node_attn, edge_atten=1 - T)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(batch, get_rep=True, node_attn=self.node_attn, edge_atten=1 - T)
            # clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            return causal_pred, causal_rep
        else:
            return causal_pred
        


class MACRO(nn.Module):

    def __init__(self,
                 input_dim,
                 out_dim,

                 noise=1.0,
                 temp=1.0,
                 node_attn='none',
                 max_iter=30,
                 gumbel_samples=5,

                 edge_dim=-1,
                 emb_dim=300,
                 num_layers=5,
                 ratio=0.25,
                 gnn_type='gin',
                 virtual_node=True,
                 residual=False,
                 drop_ratio=0.5,
                 JK="last",
                 graph_pooling="mean",
                 c_dim=-1,
                 c_in="raw",
                 c_rep="rep",
                 c_pool="add",
                 s_rep="rep",
                 sigma_len=3):
        super(MACRO, self).__init__()
        ### GNN to generate node embeddings
        if gnn_type.lower() == "le":
            self.gnn_encoder = LeGNN(in_channels=input_dim,
                                     hid_channels=emb_dim,
                                     num_layer=num_layers,
                                     drop_ratio=drop_ratio,
                                     num_classes=out_dim,
                                     edge_dim=edge_dim)
        else:
            if virtual_node:
                self.gnn_encoder = GNN_node_Virtualnode(num_layers,
                                                        emb_dim,
                                                        input_dim=input_dim,
                                                        JK=JK,
                                                        drop_ratio=drop_ratio,
                                                        residual=residual,
                                                        gnn_type=gnn_type,
                                                        edge_dim=edge_dim)
            else:
                self.gnn_encoder = GNN_node(num_layers,
                                            emb_dim,
                                            input_dim=input_dim,
                                            JK=JK,
                                            drop_ratio=drop_ratio,
                                            residual=residual,
                                            gnn_type=gnn_type,
                                            edge_dim=edge_dim)
        self.ratio = ratio
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.s_rep = s_rep
        self.c_rep = c_rep
        self.c_pool = c_pool

        # predictor based on the decoupled subgraph
        self.pred_head = "spu" if s_rep.lower() == "conv" else "inv"
        self.c_dim = emb_dim if c_dim < 0 else c_dim
        self.c_in = c_in
        self.c_input_dim = input_dim if c_in.lower() == "raw" else emb_dim
        self.classifier = GNN(gnn_type=gnn_type,
                              input_dim=self.c_input_dim,
                              num_class=out_dim,
                              num_layer=num_layers,
                              emb_dim=self.c_dim,
                              drop_ratio=drop_ratio,
                              virtual_node=virtual_node,
                              graph_pooling=graph_pooling,
                              residual=residual,
                              JK=JK,
                              pred_head=self.pred_head,
                              edge_dim=edge_dim)

        # self.gnn_encoder.reset_parameters()
        # self.classifier.reset_parameters()

        self.log_sigmas = nn.Parameter(torch.zeros(sigma_len))
        self.log_sigmas.requires_grad_(True)

        # GSTOPR args
        self.noise = noise
        self.temp = temp
        self.node_attn = node_attn
        self.max_iter = max_iter
        self.gumbel_samples = gumbel_samples



    def forward(self, batch, eps=1e-20, return_data="pred", return_spu=False, debug=False):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        
        # get edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep)#.view(-1) # edge_attn, \in R (e, 1)

        atts = (pred_edge_weight - pred_edge_weight.mean()) / pred_edge_weight.std() # norm
        def sample_gumbel(shape, eps=1e-20):
            u = torch.rand(shape)
            return -torch.log(-torch.log(u + eps) + eps)
        g = sample_gumbel(atts.shape).to(atts.device) 

        d = atts + g * self.noise if self.training else atts # (e, 1)
        s_min = atts.min() #if degrees[i] > 0 else g.min()
        s_max = atts.max() #if degrees[i] > 0 else g.max()
        D = torch.cat([d -s_min, s_max - d], dim=-1) # (b, e, 2)
        logT = -D / self.temp
        row_sum = torch.tensor([[logT.shape[0] * (1-self.ratio), logT.shape[0] * self.ratio]], device=atts.device) # (1, 2)
        for i in range(self.max_iter):
            logT = logT - torch.logsumexp(logT, dim=-1, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=-2, keepdim=True) # row norm
            logT = logT + torch.log(row_sum + eps)
        T = logT.exp()
        # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1

        T = T[:, [1]]



        # sample edge weight
        # node_batch = batch.batch # (n,)
        # edge_batch = node_batch[batch.edge_index[0]] # (e,)
        # batch_size = edge_batch.max().long().item() + 1 # b
        # degrees = degree(edge_batch).long() # (b,)
        # max_len = degrees.max().item() # s
        # atts = torch.zeros(batch_size, max_len, 1, device=device) # (b, s, 1)

        # def sample_gumbel(shape, eps=1e-20):
        #     u = torch.rand(shape)
        #     return -torch.log(-torch.log(u + eps) + eps)

        # g = sample_gumbel(atts.shape).to(device) # gumbel noise
        # for i in range(batch_size):
        #     atti = pred_edge_weight[edge_batch == i] # (ei, 1)
        #     atti = (atti - atti.mean()) / atti.std() # norm
        #     atts[i, :degrees[i]] = atti
        #     atts[i, degrees[i]:] = atti.min() #- atti.std() # sequence padding
        #     g[i, degrees[i]:] = g.min() # sequence padding

        # d = atts + g * self.noise if self.training else atts # (b, e, 1)
        # # sinkhorn soft topr
        # # D = torch.cat([d, 1 - d], dim=-1) # (b, e, 2)
        # # s_min = atts.min(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.min()
        # # s_max = atts.max(dim=1, keepdim=True)[0] #if degrees[i] > 0 else g.max()
        # D = torch.cat([d, 1 - d], dim=-1) # (b, e, 2)
        # logT = -D / self.temp
        # degrees = degrees.unsqueeze(-1) # (b, 1)
        # row_sum = torch.stack((logT.shape[1] - self.ratio * degrees + eps, self.ratio * degrees + eps), dim=-1) # (b, 1, 2)
        # for i in range(self.max_iter):
        #     logT = logT - torch.logsumexp(logT, dim=2, keepdim=True) # col norm
        #     logT = logT - torch.logsumexp(logT, dim=1, keepdim=True) # row norm
        #     logT = logT + torch.log(row_sum)
        # T = logT.exp()
        # # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1

        # Ts = []
        # for i in range(batch_size):
        #     Ti = T[i, :degrees[i], [1]] # T_res
        #     # Ti = T[i, :degrees[i], 1] # T_res
        #     Ts.append(Ti)
        # T = torch.cat(Ts, dim=0) # (e, 1)

        # obtain predictions with the classifier based on the sampled edge attentions
        # set_masks(self.classifier, T, batch.edge_index, apply_sigmoid=False)
        # set_masks(T, self.classifier, apply_sigmoid=False)
        if self.c_in == 'feat':
            batch.x = h 
        causal_pred, causal_rep = self.classifier(batch, get_rep=True, node_attn=self.node_attn, edge_atten=T)
        # clear_masks(self.classifier)
        # return causal_pred
    
        # whether to return the \hat{G_s} for further use
        if return_spu:
            # set_masks(self.classifier, 1-T, batch.edge_index, apply_sigmoid=False)
            if self.s_rep.lower() == "conv":
                spu_pred, spu_rep = self.classifier.get_spu_pred_forward(batch, get_rep=True, node_attn=self.node_attn, edge_atten=1 - T)
            else:
                spu_pred, spu_rep = self.classifier.get_spu_pred(batch, get_rep=True, node_attn=self.node_attn, edge_atten=1 - T)
            # clear_masks(self.classifier)
            causal_pred = (causal_pred, spu_pred)

        if return_data.lower() == "pred":
            return causal_pred
        elif return_data.lower() == "rep":
            return causal_pred, causal_rep
        elif return_data.lower() == "feat":
            return causal_pred, causal_rep
        else:
            return causal_pred
        
