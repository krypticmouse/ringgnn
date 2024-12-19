from torch import Tensor
from typing import Union
from torch_geometric.data import Batch

from ringsage.ringsageconv.cycle_detection import get_cycle_info


def cycle_collate_fn(batch):
    """
    Collates a list of graph data objects into a single batch, adding cycle information.

    Args:
        batch (list): A list of PyTorch Geometric Data objects representing individual graphs.

    Returns:
        torch_geometric.data.Batch: A batch object containing the collated graph data and cycle information.
    """

    # Create Batch object
    batch = Batch.from_data_list(batch)

    # Extract cycle info for the batch
    batch.cycle_info = get_cycle_info(batch)    #type: ignore

    return batch


def generate_report(
    epoch: int,
    train_loss: Union[float, Tensor],
    val_loss: Union[float, Tensor],
    val_metric: Union[float, int, Tensor],
):
    print("==" * 10)
    print(f"Epoch: {epoch}")
    print(f"Train Loss: {train_loss}")
    print(f"Val Loss: {val_loss}")
    print(f"Val Metric: {val_metric}")
    print("==" * 10)
