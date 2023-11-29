# From Discovering Invariant Rationales for Graph Neural Networks

import torch
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import global_mean_pool
from .conv_layers import LEConv
from torch_scatter import scatter_add
from torch_scatter import scatter_max
from torch_geometric.utils import degree


class SPMotifNet(torch.nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config, node_attn, c_in):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim

        self.node_emb = Linear(x_dim, hidden_size)

        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(self.n_layers):
            conv = LEConv(in_channels=hidden_size, out_channels=hidden_size)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.pool = global_mean_pool

        self.node_attn = node_attn

        self.fc_out = torch.nn.Sequential(
            Linear(hidden_size, 2*hidden_size),
            ReLU(),
            Linear(2*hidden_size, num_class)
        )

        self.conf_mlp = torch.nn.Sequential(
            Linear(hidden_size, 2*hidden_size),
            ReLU(),
            Linear(2*hidden_size, 3)
        )
        self.cq = Linear(3, 3)
        self.conf_fw = torch.nn.Sequential(
            self.conf_mlp,
            self.cq
        )

    def forward(self, x, edge_index, batch, edge_attr, edge_atten=None):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch, edge_atten=edge_atten)
        graph_x = self.pool(node_x, batch)
        return self.get_causal_pred(graph_x)

    def get_emb(self, x, edge_index, batch, edge_attr, edge_atten=None):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch, edge_atten=edge_atten)
        
        return node_x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    def get_node_reps(self, x, edge_index, edge_attr, batch, edge_atten):
        x = self.node_emb(x)
        for i, (conv, ReLU) in enumerate(zip(self.convs, self.relus)):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr, edge_atten=edge_atten)
            x = ReLU(x)
        node_x = x

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
                node_x = node_x * node_att
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
                node_x = node_x * node_att
        

        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch, edge_atten):

        node_x = self.get_node_reps(x, edge_index, edge_attr, batch, edge_atten=edge_atten)
        graph_x = self.pool(node_x, batch)
        return graph_x

    def get_causal_pred(self, causal_graph_x):
        pred = self.fc_out(causal_graph_x)
        return pred

    def get_conf_pred(self, conf_graph_x):
        pred = self.conf_fw(conf_graph_x)
        return pred

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.fc_out(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)
