"""
@Author: xjiaxin
@Date: 2022/03/23 17:26
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
        """
        使用协方差矩阵，融合特征
        :param args: f1, f2
        :return:
        """
        f1, f2 = args  # BNC
        B, num, c = f1.shape
        C = torch.matmul(f1.transpose(2, 1), f2)  # BCC
        C_col = torch.sum(C, dim=1).reshape(B, 1, c)  # B1C
        C_row = torch.sum(C, dim=2).reshape(B, 1, c)  # B1C
        alpha = 1 + torch.sigmoid(C_row)
        beta = 1 + torch.sigmoid(C_col)
        f1_newM = torch.mul(f1, alpha)
        # f1_newA = f1 + alpha

        f2_newM = torch.mul(f2, beta)
        # f2_newA = f2 + beta
        # return f1_newM, f2_newM, f1_newA, f2_newA
        # return f1_newM
        return f1_newM, f2_newM


if __name__ == '__main__':
    a = torch.rand(2, 1, 1024)
    b = torch.rand(2, 1, 1024)
    fusion = fusion()
    a, b = fusion(a, b)
    print(a.shape)
    print(b.shape)
