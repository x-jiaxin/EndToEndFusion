"""
@Date: 2022/06/16 14:45
"""
from model.DGCNN import DGCNN
from model.PointnetAttention import PointNet
import torch
from model.fusion import fusion
from torch import nn
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic
import torch.nn.functional as F


class FNet(nn.Module):
    def __init__(self, pt=PointNet(emb_dims=1024 + 64), dgcnn=DGCNN(), droput=0.0, pooling='max'):
        super(FNet, self).__init__()
        self.pt = pt
        self.dgcnn = dgcnn
        self.pooling = Pooling(pooling)
        self.fusion = fusion()
        self.fc1 = nn.Linear(1024 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)
        self.fc5 = nn.Linear(self.dgcnn.emb_dims + self.pt.emb_dims, 1024)

    def forward(self, template, source, maxIteration=2):
        template_pt = self.pooling(self.pt(template))  # 1088
        template_dg = self.pooling(self.dgcnn(template))  # 1024

        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        for i in range(maxIteration):
            est_R, est_t, source = self.get_transforme(template_pt, template_dg, source, est_R, est_t)
        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  'transformed_source': source}  # 得到两个全局特征的差值[B,feature_shape]
        return result

    def get_transforme(self, template_pt, template_dg, source, est_R, est_t):
        source_pt = self.pooling(self.pt(source))  # 1088
        source_dg = self.pooling(self.dgcnn(source))

        # Fusion
        source_pt_f, template_pt_f = self.fusion(source_pt.unsqueeze(1), template_pt.unsqueeze(1))
        source_pt = source_pt + source_pt_f.squeeze(1)
        template_pt = template_pt + template_pt_f.squeeze(1)

        source_y = torch.cat([source_pt, source_dg], dim=1)
        y1 = F.relu(self.fc5(source_y))
        template_y = torch.cat([template_pt, template_dg], dim=1)
        y2 = F.relu(self.fc5(template_y))
        y = torch.cat([y1, y2], dim=1)
        pose_8d = F.relu(self.fc1(y))
        pose_8d = F.relu(self.fc2(pose_8d))
        pose_8d = F.relu(self.fc3(pose_8d))
        pose_8d = self.fc4(pose_8d)  # [B,8]
        # 得到R
        pose_8d = PCRNetTransform.create_pose_8d(pose_8d)
        R_qe = pose_8d[:, 0:4]
        D_qe = pose_8d[:, 4:]
        # get current R and t
        est_R_temp, est_t_temp = dual_quat_to_extrinsic(R_qe, D_qe)
        # update rotation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        est_R = torch.bmm(est_R_temp, est_R)
        # Ps' = est_R * Ps + est_t
        source = PCRNetTransform.quaternion_transform2(source, pose_8d, est_t_temp)
        return est_R, est_t, source


if __name__ == '__main__':
    device = torch.device('cuda')
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
