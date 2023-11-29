# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/example.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear, Sigmoid
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import BatchNorm, global_mean_pool
from .conv_layers import PNAConvSimple

from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_geometric.utils import degree

class ToFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.float()

class PNA(torch.nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, node_attn, c_in='raw'):
        super().__init__()
        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']
        self.dropout_p = model_config['dropout_p']
        self.edge_attr_dim = edge_attr_dim


        if model_config.get('atom_encoder', False) and c_in == 'raw':
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and model_config.get('use_edge_attr', True):
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = torch.nn.Sequential(ToFloat(), Linear(x_dim, hidden_size))
            if edge_attr_dim != 0 and model_config.get('use_edge_attr', True):
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        aggregators = model_config['aggregators']
        scalers = ['identity', 'amplification', 'attenuation'] if model_config['scalers'] else ['identity']
        deg = model_config['deg']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        if model_config.get('use_edge_attr', True):
            in_channels = hidden_size * 2 if edge_attr_dim == 0 else hidden_size * 3
        else:
            in_channels = hidden_size * 2

        for _ in range(self.n_layers):
            conv = PNAConvSimple(in_channels=in_channels, out_channels=hidden_size, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_size))

        self.pool = global_mean_pool
        self.fc_out = Sequential(Linear(hidden_size, hidden_size//2), ReLU(),
                                 Linear(hidden_size//2, hidden_size//4), ReLU(),
                                 Linear(hidden_size//4, 1 if num_class == 2 and not multi_label else num_class))

        self.x_out = Sequential(Linear(hidden_size, hidden_size//2), ReLU(),
                                 Linear(hidden_size//2, hidden_size//4), ReLU(),
                                 Linear(hidden_size//4, 1), Sigmoid())

        self.node_attn = node_attn

    def forward(self, x, edge_index, batch, edge_attr, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            x = h + x  # residual#
            x = F.dropout(x, self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree

                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                node_att_sum[node_att_sum == 0] = 1
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                batch_rescale = node_nums / node_att_sum # (b, 1)
                node_rescale = batch_rescale[batch]
                node_att = node_att * node_rescale
                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)
                # print(node_att)
                # print(degree(batch))

                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                # assert 0, node_att_sum
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                node_att_sum[node_att_sum == 0] = 1 # empty graph
                batch_rescale = node_nums / node_att_sum # (b, 1)
                # print(batch_rescale)
                node_rescale = batch_rescale[batch]
                node_att = node_att * node_rescale
                # assert 0
                x = x * node_att

        x = self.pool(x, batch) # weighted mean pool
        return self.fc_out(x) # clf logits

    def get_emb(self, x, edge_index, batch, edge_attr, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            x = h + x  # residual#
            x = F.dropout(x, self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree
                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                batch_rescale = node_nums / node_att_sum # (b, 1)
                node_rescale = batch_rescale[batch]
                
                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)
                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                batch_rescale = node_nums / node_att_sum # (b, 1)
                node_rescale = batch_rescale[batch]

                x = x * node_att

        return x

    def get_graph_repr(self, x, edge_index, batch, edge_attr, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_atten)))
            x = h + x  # residual#
            x = F.dropout(x, self.dropout_p, training=self.training)

        if edge_atten is not None:
            if self.node_attn == 'none':
                pass
            elif self.node_attn == 'mean':
                node_att_sum = scatter_add(edge_atten, edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_degree = scatter_add(torch.ones_like(edge_atten), edge_index[1], dim=0, dim_size=x.size(0)) + scatter_add(torch.ones_like(edge_atten), edge_index[0], dim=0, dim_size=x.size(0))
                node_degree[node_degree == 0] = 1
                node_att = node_att_sum / node_degree
                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                batch_rescale = node_nums / node_att_sum # (b, 1)
                node_rescale = batch_rescale[batch]
                
                x = x * node_att
            elif self.node_attn == 'max':
                node_att0, _ = scatter_max(edge_atten, edge_index[0], dim=0, dim_size=x.size(0))
                node_att1, _ = scatter_max(edge_atten, edge_index[1], dim=0, dim_size=x.size(0))
                node_att = torch.maximum(node_att0, node_att1)
                batch_size = batch.max().item() + 1
                node_att_sum = scatter_add(node_att, batch, dim=0, dim_size=batch_size)
                node_nums = scatter_add(torch.ones_like(node_att), batch, dim=0, dim_size=batch_size)
                batch_rescale = node_nums / node_att_sum # (b, 1)
                node_rescale = batch_rescale[batch]

                x = x * node_att

        x = self.pool(x, batch)
        return x
    
    
    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    
