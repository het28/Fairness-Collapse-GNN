import math
import torch
from tqdm import tqdm
from fame import FAME, A_FAME
from enhanced_fame import EnhancedFAME, EnhancedAFAME
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential, Module
from calculate_fairness import calculate_fairness
from torch_geometric.data import Data as torch_geometric_Data

try:
    from torch_geometric.nn.norm import PairNorm as PyGPairNorm
except ImportError:
    PyGPairNorm = None


class _PairNormFallback(Module):
    """Simple PairNorm: center then scale by sqrt(mean(||x_i^c||^2))."""

    def __init__(self, scale=1.0, eps=1e-5):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x, batch=None, batch_size=None):
        x_centered = x - x.mean(dim=0, keepdim=True)
        row_norm_sq = (x_centered.pow(2).sum(dim=1) + self.eps).mean()
        x_scaled = self.scale * x_centered / (math.sqrt(row_norm_sq) + self.eps)
        return x_scaled

def _apply_gat_balanced_init(module, num_conv_layers):
    """
    Balanced init for GAT (Mustafa et al., NeurIPS 2023): scale all parameters
    by 1/sqrt(L) so gradient flow is balanced across layers. Applied to
    conv1, convs, conv2, res_proj1 (if present), and all GATConv/Linear params.
    """
    scale = 1.0 / math.sqrt(max(1, num_conv_layers))
    for name, param in module.named_parameters():
        param.data.mul_(scale)


class GNN(torch.nn.Module):
    def __init__(
        self, 
        data: torch_geometric_Data, 
        model: str = "GCN", 
        fame: bool = False, 
        enhanced: bool = False,
        sens_attribute: torch.Tensor = None, 
        layers: int = 2, 
        hidden: int = 16, 
        dropout: float = 0.5,
        residual: bool = False,
        pair_norm: bool = False,
        gat_balanced_init: bool = False,
        num_classes: int = None,
    ):
        super(GNN, self).__init__()
        self.residual = residual
        self.pair_norm = pair_norm
        self._model_type = model
        self._fame = fame
        self._num_layers = layers
        out_channels = num_classes if num_classes is not None else getattr(data, 'num_classes', 2)
        in_dim = data.num_node_features

        if model == "GCN":    
            self.convs = torch.nn.ModuleList()

            if fame:
                if enhanced:
                    self.conv1 = EnhancedFAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(EnhancedFAME(hidden, hidden, sens_attribute))
                    self.conv2 = EnhancedFAME(hidden, out_channels, sens_attribute)
                else:
                    self.conv1 = FAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(FAME(hidden, hidden, sens_attribute))
                    self.conv2 = FAME(hidden, out_channels, sens_attribute)
            else:
                self.conv1 = GCNConv(data.num_node_features, hidden)
                for i in range(layers - 1):
                    self.convs.append(GCNConv(hidden, hidden))
                self.conv2 = GCNConv(hidden, out_channels)

        elif model == "GAT":    
            self.convs = torch.nn.ModuleList()

            if fame:
                if enhanced:
                    self.conv1 = EnhancedAFAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(EnhancedAFAME(hidden, hidden, sens_attribute))
                    self.conv2 = EnhancedAFAME(hidden, out_channels, sens_attribute)
                else:
                    self.conv1 = A_FAME(data.num_node_features, hidden, sens_attribute)
                    for i in range(layers - 1):
                        self.convs.append(A_FAME(hidden, hidden, sens_attribute))
                    self.conv2 = A_FAME(hidden, out_channels, sens_attribute)
            else:
                self.conv1 = GATConv(data.num_node_features, hidden)
                for i in range(layers - 1):
                    self.convs.append(GATConv(hidden, hidden))
                self.conv2 = GATConv(hidden, out_channels)

        elif model == "GIN":
            # Vanilla GIN only (no FAME in baseline repo). Each layer uses MLP -> GINConv.
            self.convs = torch.nn.ModuleList()
            def _gin_mlp(in_ch, out_ch):
                return Sequential(
                    Linear(in_ch, 2 * out_ch),
                    BatchNorm1d(2 * out_ch),
                    ReLU(),
                    Linear(2 * out_ch, out_ch),
                )
            self.conv1 = GINConv(_gin_mlp(data.num_node_features, hidden), train_eps=True)
            for i in range(layers - 1):
                self.convs.append(GINConv(_gin_mlp(hidden, hidden), train_eps=True))
            self.conv2 = GINConv(_gin_mlp(hidden, out_channels), train_eps=True)

        else:
            raise ValueError(f"Unknown model: {model}. Choose from 'GCN', 'GAT', 'GIN'.")

        self.dropout = dropout
        if residual:
            self.res_proj1 = Linear(in_dim, hidden)

        if pair_norm:
            self.pair_norm_layer = PyGPairNorm(scale=1.0) if PyGPairNorm is not None else _PairNormFallback(scale=1.0)
        else:
            self.pair_norm_layer = None

        if model == "GAT" and not fame and gat_balanced_init:
            num_conv_layers = 1 + (layers - 1) + 1
            _apply_gat_balanced_init(self, num_conv_layers)

    def _maybe_pair_norm(self, x):
        if self.pair_norm_layer is not None:
            return self.pair_norm_layer(x)
        return x

    def forward(self, x, edge_index, *args, **kwargs):
        if self.residual:
            x0 = x
            x = self.conv1(x, edge_index) + self.res_proj1(x0)
            x = self._maybe_pair_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            for conv in self.convs:
                x = conv(x, edge_index) + x
                x = self._maybe_pair_norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = self.conv1(x, edge_index)
            x = self._maybe_pair_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            for conv in self.convs:
                x = conv(x, edge_index)
                x = self._maybe_pair_norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

def train(
    model: torch.nn.Module, 
    data: torch_geometric_Data, 
    optimizer: torch.optim.Optimizer, 
    epochs: int,
    loss_fn: torch.nn.Module = torch.nn.NLLLoss(),
):
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

        loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                val_out = model(data.x, data.edge_index)
                val_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask])
                print(f'Epoch {epoch} | Loss: {loss.item()} | Validation Loss: {val_loss.item()}')
            model.train()


@torch.no_grad()
def test(
    model: torch.nn.Module, 
    data: torch_geometric_Data, 
    sens_attributes: torch.Tensor,
    verbose: bool = False,
):
    with torch.inference_mode():
      out = model(data.x, data.edge_index)

    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    
    predictions = out.argmax(dim=1)
    predictions = predictions.to('cpu') 

    fairness_metrics = calculate_fairness(data, predictions, sens_attributes, verbose)
    fairness_metrics['Accuracy'] = accuracy

    return fairness_metrics