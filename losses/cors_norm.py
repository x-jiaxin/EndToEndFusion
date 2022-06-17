import torch
from torch import nn
from torch.nn import functional as F


def NormLoss(a, b):
    # sqrt = torch.norm(matrix)
    # return sqrt * sqrt
    return F.mse_loss(a, b, size_average=False)


class Corsnorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return NormLoss(a, b)
