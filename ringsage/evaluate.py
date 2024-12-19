import torch

from torch import nn
from tqdm import tqdm
from typing import Any

from .types import Task
from .schemas import EvaluationReport


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    eval_metric: Task = Task.GRAPH_REGRESSION,
    device: str = "cuda"
) -> EvaluationReport:
    """
    Evaluate the performance of a GNN model on a given dataset.

    Args:
        model (nn.Module): The GNN model to evaluate.
        loader (Any): The data loader for the dataset.
        eval_metric (Task, optional): The type of evaluation metric to use. Defaults to Task.GRAPH_REGRESSION.

    Returns:
        tuple: A tuple containing the average loss and the evaluation metric value.
    """

    model.eval()
    model = model.cuda() if device == "cuda" else model.cpu()

    pbar = tqdm(enumerate(loader), total=len(loader))

    loss = 0
    metric = 0
    preds = []

    with torch.no_grad():
        for idx, chunk in pbar:
            # Get the data attributes from batch
            x, edge_index, batch, y = chunk.x, chunk.edge_index, chunk.batch, chunk.y

            # Tranfer the data to device if needed
            x = x.cuda() if device == "cuda" else x.cpu()
            edge_index = edge_index.cuda() if device == "cuda" else edge_index.cpu()
            batch = batch.cuda() if device == "cuda" else batch.cpu()
            edge_attr = chunk.edge_attr.cuda() if device == "cuda" else chunk.edge_attr.cpu()
            cycle_info = chunk.cycle_info
            y = y.cuda() if device == "cuda" else y.cpu()

            # Run batch prediction for GNN and calculate the loss for the prediction
            pred = model(x.float(), edge_index, batch, edge_attr=edge_attr, cycle_info=cycle_info)
            loss += model.loss(pred.flatten(), y)

            # Calcuate the validation metric based on the task type
            if eval_metric == Task.GRAPH_CLASSIFICATION:
                metric += (torch.argmax(pred, dim=1) == y).sum()
            elif eval_metric == Task.GRAPH_REGRESSION:
                metric += torch.abs(pred.flatten() - y).mean()

            preds.append(pred)

            pbar.set_description(f"Evaluation Metric: {metric / (idx + 1)}")
            pbar.update(1)

    # Calculate average test loss and metric
    loss = loss / len(loader)
    metric = metric / len(loader)

    return EvaluationReport(
        loss=loss,
        metric=metric,
        predictions=preds,
    )
