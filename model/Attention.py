"""
@Date: 2022/06/16 14:45
"""
from math import sqrt

from torch import nn
import torch
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        Q, K, V = input
        dk = Q.shape[1]
        x = torch.matmul(Q.transpose(1, 2), K) / sqrt(dk)
        score = torch.sigmoid(x)
        output = torch.matmul(V, score)
        return output


if __name__ == '__main__':
    a = torch.rand(2, 1024, 4)
    b = torch.rand(2, 64, 4)
    attn = Attention()
    attn(b, b, a)
