# RingSAGE: Cycle-Aware Molecular Regression with Virtual Nodes

RingSAGE is a novel Graph Neural Network (GNN) architecture designed for cycle detection and molecular property prediction. By incorporating virtual nodes into the message-passing framework, RingSAGE can capture crucial cyclic information in molecules, improving the performance on tasks such as molecular regression and classification.


## Features

- Automatically detects minimal cycles in molecular graphs using NetworkX.
- Incorporates virtual nodes (via RingSAGE layers) to effectively enhance node representations.
- Provides a consistent interface for training, validation, and testing in PyTorch Geometric.
- Supports multiple GNN backbones, including GCN, GIN, GAT, and (by default) RingSAGE.
- Handles both graph classification and regression tasks.


## Installation
To install RingSAGE, simply run:

```bash
pip install ringsage
```

## Quick Start

Below is a minimal example of how you might use RingSAGE within a training script:

```python
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from ringsage.model import MoleculeRegressor
from ringsage.types import GNN, Task, Optimizer, Scheduler
from ringsage.schemas import ModelConfig
from ringsage.train import train
from ringsage.utils import cycle_collate_fn

train_dataset = ZINC("", True, split = 'train')
val_dataset = ZINC("", True, split = 'val')
test_dataset = ZINC("", True, split = 'test')

train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=cycle_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=cycle_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=cycle_collate_fn)


config = ModelConfig(
    task_type = Task.GRAPH_REGRESSION,
    num_features = train_dataset.num_node_features,
    hidden_channels = 128,
    num_classes = 1,
    gnn_depth = 4,
    gnn_module = GNN.RINGSAGE,
    scheduler = Scheduler.COSINE,
    optimizer = Optimizer.ADAM,
    num_edge_features = train_dataset.num_edge_features,
    num_node_features = train_dataset.num_node_features
)
model = MoleculeRegressor(config)

model_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {model_params}")

optimizer = model.configure_optimizers(lr=1e-3)
scheduler = model.configure_schedulers(optimizer, T_max=100)

report = train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    val_metric="regression"
)
print("Training complete! Final report:", report)
```

## Contributing
We welcome contributions! To contribute, please clone the repository and open a pull request with any improvements or fixes.

## License
This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details. 

## Citation
If you use RingSAGE in a research work, please cite the repository and any relevant papers. We appreciate your support and contribution to open-source software.
