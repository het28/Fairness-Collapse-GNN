"""
Graph augmentation for NIFTY-style training: drop edges and drop features.
"""

import torch
from torch_geometric.data import Data


def drop_edges(edge_index: torch.Tensor, drop_rate: float, training: bool = True) -> torch.Tensor:
    """Keep each edge with prob (1 - drop_rate). Returns edge_index (possibly same object if drop_rate=0)."""
    if not training or drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    keep = torch.rand(num_edges, device=edge_index.device, dtype=torch.float32) > drop_rate
    return edge_index[:, keep]


def drop_features(x: torch.Tensor, drop_rate: float, training: bool = True) -> torch.Tensor:
    """Zero out each feature entry with prob drop_rate (per element). Returns new tensor."""
    if not training or drop_rate <= 0:
        return x
    mask = (torch.rand_like(x, device=x.device) > drop_rate).float()
    return x * mask


def augmented_data(
    data,
    drop_edge_rate: float = 0.1,
    drop_feature_rate: float = 0.1,
    training: bool = True,
):
    """
    Return a new Data with same masks/labels but augmented x and edge_index.
    Drops edges and features with the given rates. Used for NIFTY-style consistency.
    """
    x_aug = drop_features(data.x, drop_feature_rate, training=training)
    edge_index_aug = drop_edges(data.edge_index, drop_edge_rate, training=training)
    return Data(
        x=x_aug,
        edge_index=edge_index_aug,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    ).to(data.x.device)
