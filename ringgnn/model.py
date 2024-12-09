import torch
import torch.nn.functional as F     # noqa

from torch import nn
from typing import Callable, Optional
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    global_add_pool,
)

from ringgnn.schemas import ModelConfig
from ringgnn.types import GNN, Scheduler, Task


class GNNMoleculeClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super(GNNMoleculeClassifier, self).__init__()

        self.convs = nn.ModuleList()

        num_features = cfg.num_features
        hidden_channels = cfg.hidden_channels
        num_classes = cfg.num_classes
        gnn_module = self.build_conv_layer(cfg.gnn_module)

        for idx in range(cfg.gnn_depth):
            input_channels = hidden_channels if idx > 0 else num_features

            if cfg.gnn_module == GNN.GIN:
                conv = GCNConv(input_channels, hidden_channels)
                self.convs.append(GINConv(conv))
            else:
                self.convs.append(gnn_module(input_channels, hidden_channels))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_classes),
        )

        self.relu = nn.ReLU()


    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.relu(conv(x, edge_index))

        x = global_add_pool(x, batch)
        x = self.mlp(x)

        return x


    def build_conv_layer(self, gnn_module):
        match gnn_module:
            case GNN.GCN:
                return GCNConv
            case GNN.GAT:
                return GATConv
            case GNN.GATv2:
                return GATv2Conv
            case _:
                return gnn_module


    def loss(self, pred, target, fn: Optional[Callable] = None):
        if fn is not None:
            return fn(pred, target)

        match self.cfg.task_type:
            case Task.GRAPH_CLASSIFICATION:
                return F.cross_entropy(pred, target)
            case Task.GRAPH_REGRESSION:
                return F.mse_loss(pred, target)


    def configure_schedulers(self, optimizer, **kwargs):
        match self.cfg.scheduler:
            case Scheduler.COSINE:
                return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
            case Scheduler.STEP:
                return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
            case Scheduler.PLATEAU:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            case _:
                raise NotImplementedError


    def configure_optimizers_and_schedulers(self):
        optimizer = self.__configure_optimizers()
        scheduler = self.__configure_schedulers()

        return optimizer, scheduler
