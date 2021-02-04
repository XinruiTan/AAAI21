import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from .Entropy import Entropy
from .Heaviside import Heaviside_A, Heaviside_B

__all__ = ['Alex_val_loss']

class Alex_val_loss(nn.Module):
    def __init__(self, thresholds, b_times):
        super(Alex_val_loss, self).__init__()
        self.thresholds = thresholds
        self.b_times = b_times
        self.entropy = Entropy()
        self.heaviside_a = Heaviside_A()
        self.heaviside_b = Heaviside_B()

    def forward(self, b_tuple, target):
        e_1 = self.entropy(b_tuple[0])
        e_2 = self.entropy(b_tuple[1])
        e_3 = self.entropy(b_tuple[2])
        g_1 = self.heaviside_a(self.thresholds[0] - e_1)
        g_2 = self.heaviside_a(self.thresholds[1] - e_2) * self.heaviside_b(e_1 - self.thresholds[0])
        g_3 = self.heaviside_b(e_1 - self.thresholds[0]) * self.heaviside_b(e_2 - self.thresholds[1])
        g_1.requires_grad = False
        g_2.requires_grad = False
        g_3.requires_grad = False
        val_time = self.b_times[0] * g_1 + self.b_times[1] * g_2 + self.b_times[2] * g_3
        val_time = torch.mean(val_time)
        CE_1 = F.cross_entropy(b_tuple[0], target, reduction='none', ignore_index=-100)
        CE_2 = F.cross_entropy(b_tuple[1], target, reduction='none', ignore_index=-100)
        CE_3 = F.cross_entropy(b_tuple[2], target, reduction='none', ignore_index=-100)
        CE = CE_1 * g_1 + CE_2 * g_2 + CE_3 * g_3
        CE = torch.mean(CE)
        b = b_tuple[0] * g_1.unsqueeze(1) + b_tuple[1] * g_2.unsqueeze(1) + b_tuple[2] * g_3.unsqueeze(1)
        return b, CE, val_time, e_1, e_2, e_3


        

# a = torch.FloatTensor([[1, -10, 3, 4, 100], [1, 1, 1, 1, 1]])
# b = torch.LongTensor([4, 2])
# t = [0.01, 0.01]
# time = [16, 20, 24]
# # a = torch.FloatTensor([[100, -100]])
# m_1 = Alex_val_loss(thresholds=t, b_times=time)

# print(m_1(a, a, a, b))