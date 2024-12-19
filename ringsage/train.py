import torch

from torch import nn
from tqdm import tqdm

from .types import Task
from .utils import generate_report
from .schemas import TrainingReport


def train(
    model: nn.Module,
    trainloader,
    valloader,
    optimizer,
    scheduler,
    epochs: int,
    device: str = "cuda",
    val_metric: str = "classification",
):
    """
    Standard training loop to training model over ZINC dataset.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    trainloader : DataLoader
        DataLoader for training dataset.
    valloader : DataLoader
        DataLoader for validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer to use for training.
    epochs : int
        Number of epochs to train for.
    device : str, optional
        Device to train on, by default "cpu".
    val_metric : str, optional
        Metric to use for validation, by default "classification".
    """
    metric = 0
    val_loss = 0

    train_epoch_losses = []
    val_epoch_metrics = []

    model = model.cuda() if device == "cuda" else model.cpu()

    for epoch in range(epochs):
        train_loss = 0
        step_loss = 0

        model.train()

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Step {0}")

        for idx, chunk in pbar:
            optimizer.zero_grad()

            # Get the data attributes from batch
            x, edge_index, batch, y = chunk.x, chunk.edge_index, chunk.batch, chunk.y

            # Tranfer the data to device if needed
            x = x.cuda() if device == "cuda" else x.cpu()
            edge_index = edge_index.cuda() if device == "cuda" else edge_index.cpu()
            batch = batch.cuda() if device == "cuda" else batch.cpu()
            y = y.cuda() if device == "cuda" else y.cpu()
            cycle_info = chunk.cycle_info
            edge_attr = chunk.edge_attr.cuda() if device == "cuda" else chunk.edge_attr.cpu()

            # Run batch prediction for GNN and calculate the loss for the prediction
            pred = model(x.float(), edge_index, batch, edge_attr=edge_attr, cycle_info=cycle_info)
            loss = model.loss(pred.flatten(), y)

            loss.backward()         # Calculate Gradients of params wrt loss
            optimizer.step()        # Update GNN parameters
            scheduler.step()        # Update LR for the run

            train_loss += loss.item()
            step_loss += loss.item()

            pbar.set_description(f"Step {idx} | Train Loss: {(train_loss / (idx + 1))}")
            pbar.update(1)

        # Calculate the average training loss value
        train_loss = train_loss / len(trainloader)
        train_epoch_losses.append(train_loss)

        model.eval()
        pbar = tqdm(enumerate(valloader), total=len(valloader), desc=f"Epoch {epoch}")

        val_loss = 0
        metric = 0

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
                loss = model.loss(pred.flatten(), y)

                val_loss += loss.item()

                # Calcuate the validation metric based on the task type
                if val_metric == Task.GRAPH_CLASSIFICATION:
                    metric += (torch.argmax(pred, dim=1) == y).sum()
                elif val_metric == Task.GRAPH_REGRESSION:
                    metric += torch.abs(pred.flatten() - y).mean()

                pbar.set_description(f"Evaluation Metric: {metric / (idx + 1)}")
                pbar.update(1)

        # Calculate average validation loss and metric
        val_loss = val_loss / len(valloader)
        metric = metric / len(valloader)
        val_epoch_metrics.append(metric)

        # Print the values for loss and metrics
        generate_report(epoch, train_loss, val_loss, metric)

    return TrainingReport(
        train_loss=train_epoch_losses[-1],
        val_loss=val_loss,
        val_metric=metric,
        train_loss_per_epoch=train_epoch_losses,
        val_metric_per_epoch=val_epoch_metrics,
    )
