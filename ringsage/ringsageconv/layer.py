import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


class RingSage(MessagePassing):
    def __init__(self, in_channels, out_channels, vn_dim, edge_dim, normalize=True, bias=True, **kwargs):
        """
        RingSage implementation for graph neural networks.

        Args:
            in_channels (int): Size of each input sample.
            out_channels (int): Size of each output sample.
            vn_dim (int): Dimension of virtual node features.
            edge_dim (int): Dimension of edge features.
            normalize (bool, optional): If set to :obj:`True`, output features will be `L_2`-normalized. (default: True)
            bias (bool, optional): If set to False, the layer will not learn an additive bias. (default: True)
            **kwargs (optional): Additional arguments of `torch_geometric.nn.conv.MessagePassing`.
        """
        super(RingSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vn_dim = vn_dim
        self.edge_dim = edge_dim
        self.normalize = normalize

        # Linear Layers for Message Passing
        self.lin_l = nn.Linear(in_channels + edge_dim, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        # Linear layers for Virtual Node updates
        self.lin_vn_l = nn.Linear(vn_dim, in_channels, bias=False)
        self.lin_vn_r = nn.Linear(in_channels, in_channels, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_vn_l.reset_parameters()
        self.lin_vn_r.reset_parameters()


    def forward(self, x, edge_index, edge_attr, cycle_info, size=None, vn=None):
        """
        Forward pass of the RingSage layer.

        Args:
            x (Tensor): Input node features of shape :obj:`[num_nodes, in_channels]`.
            edge_index (Tensor): Graph connectivity in COO format with shape :obj:`[2, num_edges]`.
            edge_attr (Tensor): Edge features of shape :obj:`[num_edges, edge_dim]`.
            cycle_info (dict): Dictionary containing cycle information.
            size (tuple, optional): Graph size. (default: :obj:`None`)
            vn (Tensor, optional): Virtual node features of shape :obj:`[num_cycles, vn_dim]`.

        Returns:
            Tuple[Tensor, Tensor]: Updated node and virtual node features.
        """
        x_out = None
        out_vn = None
        num_cycles = len(cycle_info)

        if num_cycles > 0:
            if vn is None:
                vn = nn.Parameter(torch.zeros(num_cycles, self.in_channels))

            # Construct custom edge_index for propagating node features to virtual nodes
            vn_edge_index = self.construct_vn_edge_index(cycle_info, x.size(0))

            # Set correct device
            vn_edge_index = vn_edge_index.to(x.device)
            vn = vn.to(x.device)

            # Update virtual node embeddings using propagate
            out_vn = self.propagate(vn_edge_index, x=(x, vn), size=(x.size(0), vn.size(0))) # num_cycles x input_dim
            out_vn = self.lin_vn_l(vn) + self.lin_vn_r(out_vn)

            # Update x embeddings using virtual node embeddings
            x_out = x + self.distribute_vn_to_nodes(out_vn, cycle_info, x.size(0))

        # Regular GraphSAGE update
        out = self.propagate(edge_index, x=(x, x), edge_attr=edge_attr, size=size)
        out = self.lin_l(out) + self.lin_r(x_out)

        if self.normalize:
            # Normalize embedding with L2 Norm
            out = F.normalize(out, p=2, dim=-1)

            assert out_vn is not None, "Virtual node embeddings are not updated."
            out_vn = F.normalize(out_vn, p=2, dim=-1)
        return out, out_vn


    def construct_vn_edge_index(self, cycle_info, num_nodes):
        """
        Construct custom edge index for propagating node features to virtual nodes.

        Args:
            cycle_info (dict): Dictionary containing cycle information.
            num_nodes (int): Number of nodes in the graph.

        Returns:
            Tensor: Custom edge index for virtual node propagation.
        """

        edge_list = []
        for vn_idx, nodes in cycle_info.items():
            # Create edges between virtual node and its nodes
            for node in nodes:
                edge_list.append([node, vn_idx])

        # Convert to tensor
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        return edge_index


    def distribute_vn_to_nodes(self, vn_emb, cycle_info, num_nodes):
        """
        Distributes virtual node embeddings to their corresponding nodes.

        Args:
            vn_emb (torch.Tensor): Embeddings of virtual nodes.
            cycle_info (dict): Dictionary containing cycle information.
            num_nodes (int): Total number of nodes in the graph.

        Returns:
            torch.Tensor: Node embeddings updated with virtual node information.
        """
        vn_sum = torch.zeros(num_nodes, vn_emb.size(1), device=vn_emb.device)
        vn_count = torch.zeros(num_nodes, 1, device=vn_emb.device)
        for vn_idx, nodes in cycle_info.items():
            # Sum virtual node embeddings for each node
            vn_sum[nodes] += vn_emb[vn_idx]

            # Count number of virtual nodes connected to each node
            vn_count[nodes] += 1

        # Average the embedding for each node
        return vn_sum / (vn_count + 1e-6)


    def message(self, x_j, edge_attr=None):
        """
        Message function for the RingSage layer.

        Args:
            x_j (Tensor): Neighbor node features of shape :obj:`[num_edges, in_channels]`.
            edge_attr (Tensor, optional): Edge features of shape :obj:`[num_edges, edge_dim]`.

        Returns:
            Tensor: Concatenated node and edge features of shape :obj:`[num_edges, in_channels + edge_dim]`.
        """
        if edge_attr is not None:
            # Concatenate node and edge features if edge features are available
            return torch.cat([x_j, edge_attr.unsqueeze(-1)], dim=-1)

        return x_j


    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """
        Aggregate function for the RingSage layer.

        Args:
            inputs (Tensor): Input features of shape :obj:`[num_nodes, in_channels]`.
            index (LongTensor): Node indices holding the same :obj:`num_edges` edges in the same order as `inputs`.

        Returns:
            Tensor: Aggregated node features of shape :obj:`[num_nodes, in_channels]`.
        """
        node_dim = self.node_dim

        # Perform mean aggregation across nodes
        return torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean')
