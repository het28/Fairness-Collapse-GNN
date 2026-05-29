import torch
from texttable import Texttable

def set_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon GPU
    return torch.device('cpu')

def print_metrics(metrics):
    table = Texttable()
    table.add_row(["Metric", "Value"])
    for key, value in metrics.items():
        table.add_row([key, value])
    print(table.draw())
    