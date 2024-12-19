import torch
import torch.nn.functional as F     # noqa

from torch import nn
from typing import Callable, Optional
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_mean_pool,
)

from .schemas import ModelConfig
from .types import GNN, Optimizer, Scheduler, Task
from .ringsageconv.layer import RingSage


class MoleculeRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig):
        """
        GNN model for molecular property prediction.

        Args:
            cfg (ModelConfig): Configuration for the GNN model.
        """
        super(MoleculeRegressor, self).__init__()

        self.cfg = cfg
        self.convs = nn.ModuleList()

        num_features = cfg.num_features
        hidden_channels = cfg.hidden_channels
        num_classes = cfg.num_classes

        # Build convolutional layers based on the specified GNN module
        for idx in range(cfg.gnn_depth):
            input_channels = hidden_channels if idx > 0 else num_features

            if cfg.gnn_module == GNN.GIN:
                dense = nn.Sequential(
                    nn.Linear(input_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                self.convs.append(GINConv(dense))
            elif cfg.gnn_module == GNN.RINGSAGE:
                vn_dim = cfg.num_node_features if idx <= 1 else hidden_channels

                self.convs.append(
                    RingSage(input_channels, hidden_channels, vn_dim=vn_dim, edge_dim=cfg.num_edge_features)
                )
            else:
                gnn_module = self.build_conv_layer(cfg.gnn_module)
                self.convs.append(gnn_module(input_channels, hidden_channels))      # type: ignore

        # Dense Layer for classifier
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

        self.edge_proj = nn.Linear(num_features, hidden_channels)
        self.relu = nn.ReLU()
        self.init_weights()


    def forward(self, x, edge_index, batch, edge_attr=None, cycle_info=None):
        """
        Forward pass of the GNN model.

        Args:
            x (Tensor): Input node features of shape num_nodes, in_channels.
            edge_index (Tensor): Graph connectivity in COO format with shape 2, num_edges.
            batch (Tensor): Batch vector assigning each node to a specific graph in the batch.
            edge_attr (Tensor, optional): Edge features of shape num_edges, edge_dim.
            cycle_info (dict, optional): Dictionary containing cycle information.

        Returns:
            Tensor: Output tensor of shape [num_nodes, num_classes].
        """

        x_vn = None

        for conv in self.convs:
            match self.cfg.gnn_module:
                case GNN.GEN:
                    assert edge_attr is not None, "Edge attributes must be provided for GENConv."

                    edge_attr_t = self.edge_proj(edge_attr.float().view(-1, 1))
                    x = self.relu(conv(x, edge_index, edge_attr_t))

                case GNN.RINGSAGE:
                    x, x_vn = conv(x, edge_index, edge_attr, cycle_info, vn=x_vn)

                    x = self.relu(x)

                case _:
                    x = self.relu(conv(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x


    def build_conv_layer(self, gnn_module):
        """
        Builds the appropriate convolutional layer based on the specified GNN module.

        Args:
            gnn_module (GNN): The type of GNN module to use.

        Returns:
            type: The convolutional layer class corresponding to the specified GNN module.
        """

        match gnn_module:
            case GNN.GCN:
                return GCNConv
            case GNN.GAT:
                return GATConv
            case GNN.GEN:
                return GCNConv
            case GNN.GRAPHSAGE:
                return SAGEConv
            case GNN.RINGSAGE:
                return RingSage
            case _:
                raise NotImplementedError


    def loss(self, pred, target, fn: Optional[Callable] = None):
        """
        Calculate the loss between the predicted and target values.

        Args:
            pred (Tensor): Predicted values.
            target (Tensor): Target values.
            fn (Callable, optional): Custom loss function.

        Returns:
            Tensor: Calculated loss.
        """
        if fn is not None:
            return fn(pred, target)

        match self.cfg.task_type:
            case Task.GRAPH_CLASSIFICATION:
                return F.cross_entropy(pred, target)
            case Task.GRAPH_REGRESSION:
                return F.l1_loss(pred, target)


    def configure_optimizers(self, **kwargs):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        match self.cfg.optimizer:
            case Optimizer.ADAM:
                return torch.optim.Adam(self.parameters(), **kwargs)
            case Optimizer.SGD:
                return torch.optim.SGD(self.parameters(), **kwargs)
            case Optimizer.ADAGRAD:
                return torch.optim.Adagrad(self.parameters(), **kwargs)
            case Optimizer.RMSPROP:
                return torch.optim.RMSprop(self.parameters(), **kwargs)
            case Optimizer.ADAMW:
                return torch.optim.AdamW(self.parameters(), **kwargs)
            case _:
                raise NotImplementedError


    def configure_schedulers(self, optimizer, **kwargs):
        """
        Configure the learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to be used.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler.
        """
        match self.cfg.scheduler:
            case Scheduler.COSINE:
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
            case Scheduler.STEP:
                return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
            case Scheduler.PLATEAU:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            case _:
                raise NotImplementedError


    def init_weights(self):
        """
        Initialize the weights of the Linear Layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
