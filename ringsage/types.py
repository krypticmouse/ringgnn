from enum import Enum


class Task(str, Enum):
    """
    Task type for GNN.
    """
    GRAPH_CLASSIFICATION = 'graph_classification'
    GRAPH_REGRESSION = 'graph_regression'


class GNN(str, Enum):
    """
    Message Passing types for GNN.
    """
    GIN = 'gin'
    GCN = 'gcn'
    GAT = 'gat'
    GATv2 = 'gatv2'
    GEN = 'gen'
    RINGSAGE = 'ringsage'
    GRAPHSAGE = 'graphsage'


class Scheduler(str, Enum):
    """
    Learning rate scheduler type.
    """
    COSINE = 'cosine'
    STEP = 'step'
    PLATEAU = 'plateau'


class Optimizer(str, Enum):
    """
    Optimizer type for updating GNN parameters.
    """
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    RMSPROP = 'rmsprop'
    ADAMW = 'adamw'
