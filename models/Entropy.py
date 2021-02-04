import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 

__all__ = ['Entropy']

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        x = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        x = -1.0 * x.sum(dim=1) / np.log(2)
        return x