# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .conv_layers import GINConv, GINEConv

from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_geometric.utils import degree

class Thresh(nn.Module):
    def __init__(self, th=10):
        super().__init__()
        self.th = th
    def forward(self, x):
        x = (x / self.th).tanh() * self.th
        return x

class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, node_attn, c_in):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)

        if model_config.get('atom_encoder', False) and c_in == 'raw':
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        self.x_out = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.node_attn = node_attn
        

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree

                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)

                x = x * node_att

        return self.fc_out(self.pool(x, batch))

    def get_graph_repr(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree

                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)

                x = x * node_att
        return self.pool(x, batch)

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree

                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)

                x = x * node_att
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
