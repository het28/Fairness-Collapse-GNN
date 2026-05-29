"""
PFR (Prejudice Remover) in-processing fairness: add a discrimination-aware
regularization term to the training loss so the model does not rely on the
sensitive attribute. Works with any backbone (GCN, GAT, GIN).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .graph_augment import drop_edges


def pfr_loss_term(
    log_softmax_out: torch.Tensor,
    sens: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PFR-style penalty: squared correlation between predicted probability of
    class 1 and sensitive attribute on the given mask (e.g. train set).
    Minimizing this reduces dependence of predictions on the sensitive attribute.

    log_softmax_out: (N, 2) log probabilities
    sens: (N,) binary sensitive attribute (0/1), same device as out
    mask: (N,) boolean
    """
    # Probability of class 1 on masked nodes (ensure mask same device as tensors)
    device = log_softmax_out.device
    mask = mask.to(device) if mask.device != device else mask
    sens = sens.to(device) if sens.device != device else sens
    p = torch.exp(log_softmax_out[mask, 1]).float().squeeze()  # (n_train,)
    s = sens[mask].float().squeeze()  # (n_train,)
    if p.numel() < 2 or s.numel() < 2:
        return p.new_tensor(0.0)
    p = p - p.mean()
    s = s - s.mean()
    cov = (p * s).mean()
    std_p = p.std() + eps
    std_s = s.std() + eps
    if std_p < eps or std_s < eps:
        return p.new_tensor(0.0)
    corr = cov / (std_p * std_s)
    # Squared correlation so it's non-negative; minimize it
    return corr.pow(2)


def train_with_pfr(
    model: torch.nn.Module,
    data,  # PyG Data with x, edge_index, y, train_mask, val_mask
    sens_attributes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    pfr_lambda: float = 0.1,
    drop_edge_rate: float = 0.0,
    loss_fn: torch.nn.Module = None,
):
    """
    Training loop that adds PFR (Prejudice Remover) regularization to the
    classification loss. When drop_edge_rate > 0, training forward uses
    randomly dropped edges (anti-oversmoothing). Val uses full graph.
    """
    if loss_fn is None:
        loss_fn = torch.nn.NLLLoss()
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        edge_index = drop_edges(data.edge_index, drop_edge_rate, training=model.training)
        out = model(data.x, edge_index)
        loss_nll = loss_fn(out[data.train_mask], data.y[data.train_mask])
        pfr_loss = pfr_loss_term(out, sens_attributes, data.train_mask)
        loss = loss_nll + pfr_lambda * pfr_loss
        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} (NLL: {loss_nll.item():.4f}, PFR: {pfr_loss.item():.4f}) | Val Loss: {val_loss.item():.4f}"
                )
            model.train()
