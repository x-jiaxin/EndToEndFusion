import torch
import torch.nn as nn
import torch.nn.functional as F
from operations.transform_functions import PCRNetTransform


def frobeniusNormLoss(predicted, igt):
    """ |predicted*igt - I| (should be 0) """
    # predicted:[B,4,4]
    # igt;[B,4,4]
    error = predicted.matmul(igt)
    I = torch.eye(4).to(error).view(1, 4, 4).expand(error.size(0), 4, 4)
    return F.mse_loss(error, I, size_average=True) * 16
    # return F.mse_loss(predicted, igt, size_average=True) * 9


def NormLoss(matrix):
    sqrt = torch.norm(matrix)
    return sqrt * sqrt


class FrobeniusNormLoss(nn.Module):
    def __init__(self):
        super(FrobeniusNormLoss, self).__init__()

    def forward(self, predicted, igt):
        return frobeniusNormLoss(predicted, igt)


if __name__ == '__main__':
    source = torch.rand(10, 1024, 3)
    igt = torch.rand(10, 7)
    output = torch.rand(10, 4, 4)
    identity = torch.eye(3).to(source).view(1, 3, 3).expand(source.size(0), 3, 3).contiguous()
    est_R = PCRNetTransform.quaternion_rotate(identity, igt).permute(0, 2, 1)
    est_t = PCRNetTransform.get_translation(igt).view(-1, 1, 3)
    igt = PCRNetTransform.convert2transformation(est_R, est_t)
    print(igt.shape)
    loss_val = FrobeniusNormLoss()(output, igt)
    print(loss_val)
