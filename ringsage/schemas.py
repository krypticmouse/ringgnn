from typing import Union
from pydantic import BaseModel
from torch_geometric.nn import MessagePassing

from .types import GNN, Optimizer, Scheduler, Task


class ModelConfig(BaseModel):
    # Data Parameters
    task_type: Task
    num_features: int
    hidden_channels: int
    num_classes: int


    # Architecture Parameters
    gnn_depth: int
    gnn_module: Union[GNN, MessagePassing]
    scheduler: Scheduler
    optimizer: Optimizer

