import torch
import inspect
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


class RingSageConv(MessagePassing):
    def __init__(self, in_channels, out_channels, vn_dim, edge_dim, normalize=True, bias=False, **kwargs):
        super(RingSageConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vn_dim = vn_dim
        self.edge_dim = edge_dim
        self.normalize = normalize

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels + edge_dim, out_channels, bias=bias)

        self.lin_vn_l = nn.Linear(vn_dim, in_channels, bias=bias)
        self.lin_vn_r = nn.Linear(in_channels, in_channels, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_vn_l.reset_parameters()
        self.lin_vn_r.reset_parameters()


    def forward(self, x, edge_index, edge_attr, cycle_info, size=None, vn=None):
        num_cycles = len(cycle_info)
        if num_cycles > 0:
          if vn is None:
              vn = nn.Parameter(torch.zeros(num_cycles, self.in_channels))

          # Construct custom edge_index for propagating node features to virtual nodes
          vn_edge_index = self.construct_vn_edge_index(cycle_info, x.size(0))

          # Update virtual node embeddings using propagate
          out_vn = self.propagate(vn_edge_index, x=(x, vn), size=(x.size(0), vn.size(0))) # num_cycles x input_dim
          out_vn = self.lin_vn_l(vn) + self.lin_vn_r(out_vn)

          # Update x embeddings using virtual node embeddings
          x = x + self.distribute_vn_to_nodes(out_vn, cycle_info, x.size(0))

        # Regular GraphSAGE update
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr, size=size)
        out = self.lin_l(x) + self.lin_r(out)

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
            out_vn = F.normalize(out_vn, p=2, dim=-1)
        return out, out_vn


    def construct_vn_edge_index(self, cycle_info, num_nodes):
        edge_list = []
        for vn_idx, nodes in cycle_info.items():
            for node in nodes:
                edge_list.append([node, vn_idx])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index


    def distribute_vn_to_nodes(self, vn_emb, cycle_info, num_nodes):
        vn_sum = torch.zeros(num_nodes, vn_emb.size(1), device=vn_emb.device)
        vn_count = torch.zeros(num_nodes, 1, device=vn_emb.device)
        for vn_idx, nodes in cycle_info.items():
            vn_sum[nodes] += vn_emb[vn_idx]
            vn_count[nodes] += 1

        return vn_sum / (vn_count + 1e-6)


    def message(self, x_j, edge_attr=None):
        if edge_attr is not inspect._empty:
            return torch.cat([x_j, edge_attr.unsqueeze(-1)], dim=-1)
        else:
            return x_j


    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim
        return torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean')
