from torch import Tensor
from typing import Union, List
from pydantic import BaseModel

from .types import GNN, Optimizer, Scheduler, Task


class ModelConfig(BaseModel):
    """
    Configuration for GNN model.
    """

    task_type: Task
    num_features: int
    hidden_channels: int
    num_classes: int
    gnn_depth: int
    gnn_module: GNN
    scheduler: Scheduler
    optimizer: Optimizer
    num_edge_features: int = 1
    num_node_features: int = 1


class TrainingReport(BaseModel):
    """
    Training report for GNN model.
    """

    train_loss: Union[float, Tensor]
    val_loss: Union[float, Tensor]
    val_metric: Union[float, int, Tensor]

    train_loss_per_epoch: List
    val_metric_per_epoch: List


class EvaluationReport(BaseModel):
    """
    Evaluation report for GNN model.
    """

    loss: float
    metric: Union[float, int, Tensor]
    predictions: List
