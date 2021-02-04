import torch
import torch.nn as nn
import numpy as np

__all__ = ['Heaviside_A', 'Heaviside_B']

class Heaviside_A(nn.Module):
    def __init__(self):
        super(Heaviside_A, self).__init__()

    def forward(self, x):
        x = np.heaviside(x.detach().cpu().numpy(), 0.0)
        x = torch.from_numpy(x)
        return x

class Heaviside_B(nn.Module):
    def __init__(self):
        super(Heaviside_B, self).__init__()

    def forward(self, x):
        x = np.heaviside(x.detach().cpu().numpy(), 1.0)
        x = torch.from_numpy(x)
        return x