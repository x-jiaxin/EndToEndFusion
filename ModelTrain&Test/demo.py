"""
@Date: 2022/06/16 15:31
"""
import torch

from model.Net import FNet
from model.DGCNN import DGCNN
from model.Pointnet import PointNet

if __name__ == '__main__':
    device = torch.device('cuda:3')
    template, source = torch.rand(2, 100, 3), torch.rand(2, 100, 3)
    template = template.to(device)
    source = source.to(device)
    pn = PointNet(emb_dims=1024)
    dg = DGCNN()
    net = FNet(pt=pn, dgcnn=dg)
    net = net.to(device)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
