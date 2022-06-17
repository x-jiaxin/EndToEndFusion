"""
Author: xjiaxin
Date: 2022-04-28 19:05:00
"""

import torch
import torch.nn as nn


class fusion(nn.Module):
    """
    input:[B,N,C]
    C:[C,C]
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        f1, f2 = args  # B,N,C
        B, N, C1 = f1.shape
        _, _, C2 = f2.shape
        Cm = torch.matmul(f1.transpose(2, 1), f2)  # [B,C1,C2]
        C_col = torch.sum(Cm, dim=1).reshape(B, -1, C1)
        C_row = torch.sum(Cm, dim=2).reshape(B, -1, C2)
        alpha = torch.sigmoid(C_row)
        beta = torch.sigmoid(C_col)
        f1_newM = torch.mul(f1, 1 - alpha)
        f2_newM = torch.mul(f2, 1 - beta)
        return f1_newM, f2_newM  # B,N,C
        # return f1_newM  # B,N,C


if __name__ == '__main__':
    a = torch.randn(2, 1, 1024)
    b = torch.randn(2, 1, 1024)
    a1, b1 = fusion()(a, b)
