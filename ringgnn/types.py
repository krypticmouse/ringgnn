from enum import Enum


class Task(Enum, str):
    """Task type for the model."""
    GRAPH_CLASSIFICATION = 'graph_classification'
    GRAPH_REGRESSION = 'graph_regression'


class GNN(Enum, str):
    """GNN module to use for message passing."""
    GIN = 'gin'
    GCN = 'gcn'
    GAT = 'gat'
    GATv2 = 'gatv2'


class Scheduler(Enum, str):
    """Learning rate scheduler."""
    COSINE = 'cosine'
    STEP = 'step'
    PLATEAU = 'plateau'


class Optimizer(Enum, str):
    """Optimizer for training the model."""
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    RMSPROP = 'rmsprop'
    ADAMW = 'adamw'
