import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
# from .Entropy import Entropy
from .Heaviside import Heaviside_A, Heaviside_B

zeros = [0.0, 0.0, 0.0, 0.0, 0.0]
ones = [1.0, 1.0, 1.0, 1.0, 1.0]

__all__ = ['MSD_val_loss_cpu']


class Maximum(nn.Module):
    def __init__(self):
        super(Maximum, self).__init__()

    def forward(self, x):
        x = F.softmax(x, 1, _stacklevel=5)
        x = -1 * torch.max(x, 1).values
        return x


class MSD_val_loss_cpu(nn.Module):
    def __init__(self, thresholds=ones, b_times=zeros):
        super(MSD_val_loss_cpu, self).__init__()
        self.thresholds = [-1.0 * t for t in thresholds]
        self.b_times = b_times
        self.maximum = Maximum()
        self.heaviside_a = Heaviside_A()
        self.heaviside_b = Heaviside_B()

    def forward(self, b_tuple, target):
        e_1 = self.maximum(b_tuple[0])
        e_2 = self.maximum(b_tuple[1])
        e_3 = self.maximum(b_tuple[2])
        e_4 = self.maximum(b_tuple[3])
        e_5 = self.maximum(b_tuple[4])
        g_1 = self.heaviside_a(self.thresholds[0] - e_1).detach()
        g_2 = self.heaviside_a(self.thresholds[1] - e_2).detach() * self.heaviside_b(e_1 - self.thresholds[0]).detach()
        g_3 = self.heaviside_a(self.thresholds[2] - e_3).detach() * self.heaviside_b(e_1 - self.thresholds[0]).detach() * self.heaviside_b(e_2 - self.thresholds[1]).detach()
        g_4 = self.heaviside_a(self.thresholds[3] - e_4).detach() * self.heaviside_b(e_1 - self.thresholds[0]).detach() * self.heaviside_b(e_2 - self.thresholds[1]).detach() * self.heaviside_b(e_3 - self.thresholds[2]).detach()
        g_5 = self.heaviside_b(e_4 - self.thresholds[3]).detach() * self.heaviside_b(e_1 - self.thresholds[0]).detach() * self.heaviside_b(e_2 - self.thresholds[1]).detach() * self.heaviside_b(e_3 - self.thresholds[2]).detach()
        g_1.requires_grad = False
        g_2.requires_grad = False
        g_3.requires_grad = False
        g_4.requires_grad = False
        g_5.requires_grad = False
        val_time = self.b_times[0] * g_1 + self.b_times[1] * g_2 + self.b_times[2] * g_3 + self.b_times[3] * g_4 + self.b_times[4] * g_5
        val_time = torch.mean(val_time)
        CE_1 = F.cross_entropy(b_tuple[0], target, reduction='none', ignore_index=-100)
        CE_2 = F.cross_entropy(b_tuple[1], target, reduction='none', ignore_index=-100)
        CE_3 = F.cross_entropy(b_tuple[2], target, reduction='none', ignore_index=-100)
        CE_4 = F.cross_entropy(b_tuple[3], target, reduction='none', ignore_index=-100)
        CE_5 = F.cross_entropy(b_tuple[4], target, reduction='none', ignore_index=-100)
        CE = CE_1 * g_1 + CE_2 * g_2 + CE_3 * g_3 + CE_4 * g_4 + CE_5 * g_5
        CE = torch.mean(CE)
        b = b_tuple[0] * g_1.unsqueeze(1) + b_tuple[1] * g_2.unsqueeze(1) + b_tuple[2] * g_3.unsqueeze(1) + b_tuple[3] * g_4.unsqueeze(1) + b_tuple[4] * g_5.unsqueeze(1)
        return b, CE, val_time, e_1, e_2, e_3, e_4, e_5

# a = torch.FloatTensor([[1, -10, 3, 4, 100], [1, 1, 1, 1, 1]])
# m = Maximum()
# print(m(a))

        

# a1 = torch.FloatTensor([[0.5, 0.5], [0.7, 0.3]])
# a2 = torch.FloatTensor([[0.6, 0.4], [0.9, 0.1]])
# a3 = torch.FloatTensor([[0.8, 0.2], [0.9, 0.1]])
# a4 = torch.FloatTensor([[0.85, 0.15], [0.9, 0.1]])
# a5 = torch.FloatTensor([[0.83, 0.17], [0.9, 0.1]])
# b = torch.LongTensor([1, 0])
# t = [0.8, 0.8, 0.8, 0.8]
# time = [16, 20, 24, 28, 32]
# # a = torch.FloatTensor([[100, -100]])
# m_1 = MSD_val_loss(thresholds=t, b_times=time)
# # m_1 = MSD_val_loss()

# print(m_1((a1, a2, a3, a4, a5), b))