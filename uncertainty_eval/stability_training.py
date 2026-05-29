"""
Option A: Stability regularization (Huang–Vishnoi style).
Two forward passes with different dropout states; add consistency loss
so predictions on train nodes are stable under perturbation.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .graph_augment import drop_edges


def train_with_stability(
    model: torch.nn.Module,
    data,  # PyG Data with x, edge_index, y, train_mask, val_mask
    optimizer: torch.optim.Optimizer,
    epochs: int,
    stability_lambda: float = 0.5,
    drop_edge_rate: float = 0.0,
    loss_fn: torch.nn.Module = None,
):
    """
    Training loop with stability regularization:
    loss = NLL + stability_lambda * MSE(out1[train], out2[train])
    where out1 and out2 are two forward passes (different dropout).
    When drop_edge_rate > 0, training uses randomly dropped edges (anti-oversmoothing).
    """
    if loss_fn is None:
        loss_fn = torch.nn.NLLLoss()
    for epoch in tqdm(range(epochs), desc="Training Epochs (stability)"):
        optimizer.zero_grad()
        edge_index = drop_edges(data.edge_index, drop_edge_rate, training=model.training)
        # Two forwards → different dropout masks, same (dropped) edges
        out1 = model(data.x, edge_index)
        out2 = model(data.x, edge_index)
        loss_nll = loss_fn(out1[data.train_mask], data.y[data.train_mask])
        # Consistency on train nodes: match log-probs (or use MSE on probs)
        # Use MSE on probabilities for stability (bounded, smooth)
        p1 = torch.exp(out1[data.train_mask])
        p2 = torch.exp(out2[data.train_mask])
        stability_loss = F.mse_loss(p1, p2)
        loss = loss_nll + stability_lambda * stability_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} (NLL: {loss_nll.item():.4f}, stab: {stability_loss.item():.4f}) | Val: {val_loss.item():.4f}"
                )
            model.train()
