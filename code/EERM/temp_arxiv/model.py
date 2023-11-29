import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from nets import *

class Base(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Base, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.n = n
        self.device = device
        self.args = args

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        out = F.log_softmax(pred, dim=1)
        target = y.squeeze(1)
        loss = criterion(out, target)
        return loss

 
class GSTOPR(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(GSTOPR, self).__init__()
        if gnn == 'gcn':
            self.gnn_encoder = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.hidden_channels,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
            self.gnn = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
            self.gnn_encoder = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.hidden_channels,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
            self.gnn_encoder = GAT(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.hidden_channels,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
            self.gnn_encoder = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=args.hidden_channels,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        # elif gnn == 'gcnii':
        #     self.gnn = GCNII(in_channels=d,
        #                 hidden_channels=args.hidden_channels,
        #                 out_channels=c,
        #                 num_layers=args.num_layers,
        #                 dropout=args.dropout,
        #                 alpha=args.gcnii_alpha,
        #                 lamda=args.gcnii_lamda)
        #     self.gnn_encoder = GCNII(in_channels=d,
        #                 hidden_channels=args.hidden_channels,
        #                 out_channels=args.hidden_channels,
        #                 num_layers=args.num_layers,
        #                 dropout=args.dropout,
        #                 alpha=args.gcnii_alpha,
        #                 lamda=args.gcnii_lamda)
        
        

        self.n = n
        self.device = device
        self.gnn_net = gnn
        self.args = args

        # GSTOPR
        emb_dim = args.hidden_channels
        self.edge_att = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim * 4), nn.ReLU(), nn.Linear(emb_dim * 4, 1))
        self.r = args.r 
        self.temp = args.temp 
        self.noise = args.noise
        self.max_iter = args.max_iter

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, criterion):
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        
        h = self.gnn_encoder(x, edge_index)
        device = h.device
        
        # get edge-level attetion
        row, col = edge_index
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        atts = self.edge_att(edge_rep) # edge_attn, \in R (e, 1)
        atts = (atts - atts.mean()) / atts.std() # norm

        def sample_gumbel(shape, eps=1e-20):
            u = torch.rand(shape)
            return -torch.log(-torch.log(u + eps) + eps)

        g = sample_gumbel(atts.shape).to(device) # gumbel noise

        d = atts + g * self.noise if self.training else atts # (e, 1)
        # D = torch.cat([d, 1 - d], dim=-1) # (e, 2)
        s_max = atts.max()
        s_min = atts.min()
        D = torch.cat([d - s_min, s_max - d], dim=-1) # (e, 2)
        logT = -D / self.temp
        eps=1e-10
        row_sum = torch.tensor([atts.shape[0] * (1 - self.r) + eps, self.r * atts.shape[0] + eps], device=device).unsqueeze(0) # (1, 2)
        for i in range(self.max_iter):
            logT = logT - torch.logsumexp(logT, dim=-1, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=-2, keepdim=True) # row norm
            logT = logT + torch.log(row_sum)
        T = logT.exp()
        # T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1
        T = T[:, [1]] # (e, 1)
        # T = T[:, 1] # (e, )

        set_masks(self.gnn, T, edge_index, apply_sigmoid=False)
        out = self.gnn(x, edge_index)
        clear_masks(self.gnn)
    
        loss = self.sup_loss(y, out, criterion)
        return loss

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        
        h = self.gnn_encoder(x, edge_index)
        device = h.device
        
        # get edge-level attetion
        row, col = edge_index
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        atts = self.edge_att(edge_rep) # edge_attn, \in R (e, 1)
        atts = (atts - atts.mean()) / atts.std() # norm


        d = atts 
        D = torch.cat([d, 1 - d], dim=-1) # (e, 2)
        logT = -D / self.temp
        eps=1e-10
        row_sum = torch.tensor([atts.shape[0] * (1 - self.r) + eps, self.r * atts.shape[0] + eps], device=device).unsqueeze(0) # (1, 2)
        for i in range(self.max_iter):
            logT = logT - torch.logsumexp(logT, dim=-1, keepdim=True) # col norm
            logT = logT - torch.logsumexp(logT, dim=-2, keepdim=True) # row norm
            logT = logT + torch.log(row_sum)
        T = logT.exp()
        T = (T - T.min()) / (T.max() - T.min()) # value range 0 ~ 1
        T = T[:, [1]] # (e, 1)
        # T = T[:, 1] # (e, )

        set_masks(self.gnn, T, edge_index, apply_sigmoid=False)
        out = self.gnn(x, edge_index)
        clear_masks(self.gnn)
    
        # out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss


class Graph_Editer(nn.Module):
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, n, num_sample, k):
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p

class Model(nn.Module):
    def __init__(self, args, n, c, d, gnn, device):
        super(Model, self).__init__()
        if gnn == 'gcn':
            self.gnn = GCN(in_channels=d,
                            hidden_channels=args.hidden_channels,
                            out_channels=c,
                            num_layers=args.num_layers,
                            dropout=args.dropout,
                            use_bn=not args.no_bn)
        elif gnn == 'sage':
            self.gnn = SAGE(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
        elif gnn == 'gat':
            self.gnn = GAT(in_channels=d,
                           hidden_channels=args.hidden_channels,
                           out_channels=c,
                           num_layers=args.num_layers,
                           dropout=args.dropout,
                           heads=args.gat_heads)
        elif gnn == 'gpr':
            self.gnn = GPRGNN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        alpha=args.gpr_alpha,
                        )
        self.p = 0.2
        self.n = n
        self.device = device
        self.args = args

        self.gl = Graph_Editer(args.K, n, device)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, data, criterion):
        Loss, Log_p = [], 0
        x, y = data.graph['node_feat'].to(self.device), data.label.to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        for k in range(self.args.K):
            edge_index_k, log_p = self.gl(edge_index, self.n, self.args.num_sample, k)
            out = self.gnn(x, edge_index_k)
            loss = self.sup_loss(y, out, criterion)
            Loss.append(loss.view(-1))
            Log_p += log_p
        Loss = torch.cat(Loss, dim=0)
        Var, Mean = torch.var_mean(Loss)
        return Var, Mean, Log_p

    def inference(self, data):
        x = data.graph['node_feat'].to(self.device)
        edge_index = data.graph['edge_index'].to(self.device)
        out = self.gnn(x, edge_index)
        return out

    def sup_loss(self, y, pred, criterion):
        if self.args.rocauc or self.args.dataset in ('twitch-e', 'fb100', 'elliptic'):
            if y.shape[1] == 1:
                true_label = F.one_hot(y, y.max() + 1).squeeze(1)
            else:
                true_label = y
            loss = criterion(pred, true_label.squeeze(1).to(torch.float))
        else:
            out = F.log_softmax(pred, dim=1)
            target = y.squeeze(1)
            loss = criterion(out, target)
        return loss