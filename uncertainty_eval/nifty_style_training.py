"""
Option B: NIFTY-style training (Agarwal et al., UAI 2021).
Forward on original and augmented graph; maximize similarity between
outputs on train nodes (stability) while minimizing NLL.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .graph_augment import augmented_data, drop_edges


def train_with_nifty(
    model: torch.nn.Module,
    data,  # PyG Data with x, edge_index, y, train_mask, val_mask
    optimizer: torch.optim.Optimizer,
    epochs: int,
    sim_coeff: float = 0.5,
    drop_edge_rate: float = 0.1,
    drop_feature_rate: float = 0.1,
    anti_oversmooth_drop_edge: float = 0.0,
    loss_fn: torch.nn.Module = None,
):
    """
    NIFTY-style: loss = NLL + sim_coeff * (1 - cosine_sim(out_orig[train], out_aug[train])).
    Augmented graph: drop edges and features at drop_edge_rate, drop_feature_rate.
    When anti_oversmooth_drop_edge > 0, out_orig uses a separate dropped edge_index (anti-oversmoothing).
    """
    if loss_fn is None:
        loss_fn = torch.nn.NLLLoss()
    for epoch in tqdm(range(epochs), desc="Training Epochs (nifty)"):
        optimizer.zero_grad()
        edge_orig = drop_edges(data.edge_index, anti_oversmooth_drop_edge, training=model.training)
        out_orig = model(data.x, edge_orig)
        data_aug = augmented_data(data, drop_edge_rate, drop_feature_rate, training=True)
        out_aug = model(data_aug.x, data_aug.edge_index)
        loss_nll = loss_fn(out_orig[data.train_mask], data.y[data.train_mask])
        # Similarity on train: maximize cosine sim between logits (or probs)
        p_orig = torch.exp(out_orig[data.train_mask])
        p_aug = torch.exp(out_aug[data.train_mask])
        # Cosine similarity (1 - cos_sim) to minimize; cos_sim in [-1,1]
        cos_sim = F.cosine_similarity(p_orig, p_aug, dim=1).mean()
        sim_loss = 1.0 - cos_sim  # minimize this
        loss = loss_nll + sim_coeff * sim_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(
                    f"Epoch {epoch} | Loss: {loss.item():.4f} (NLL: {loss_nll.item():.4f}, sim_loss: {sim_loss.item():.4f}) | Val: {val_loss.item():.4f}"
                )
            model.train()
