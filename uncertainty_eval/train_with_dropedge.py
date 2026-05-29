"""
Standard training with optional DropEdge (Rong et al., ICLR 2020).
When drop_edge_rate > 0, each epoch uses a randomly dropped edge_index for the
forward pass to reduce oversmoothing. Validation uses the full graph.
"""

import torch
from tqdm import tqdm

from .graph_augment import drop_edges


def train_standard_with_dropedge(
    model: torch.nn.Module,
    data,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    drop_edge_rate: float = 0.0,
    loss_fn: torch.nn.Module = None,
):
    """
    Same as GNNs-FAME train() but when drop_edge_rate > 0, each training
    forward uses edge_index with a fraction of edges dropped. Val/test use
    full graph.
    """
    if loss_fn is None:
        loss_fn = torch.nn.NLLLoss()

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        edge_index = drop_edges(data.edge_index, drop_edge_rate, training=model.training)
        out = model(data.x, edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} | Validation Loss: {val_loss.item():.4f}"
                )
            model.train()
