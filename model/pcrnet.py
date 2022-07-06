import torch
import torch.nn as nn
from model.Pointnet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class iPCRNet(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(iPCRNet, self).__init__()
        self.source_feature = None
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)

        self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
                       nn.Linear(1024, 1024), nn.ReLU(),
                       nn.Linear(1024, 512), nn.ReLU(),
                       nn.Linear(512, 512), nn.ReLU(),
                       nn.Linear(512, 256), nn.ReLU()]

        if droput > 0.0:
            self.linear.append(nn.Dropout(droput))
        self.linear.append(nn.Linear(256, 7))

        self.linear = nn.Sequential(*self.linear)

    def forward(self, template, source, maxIteration=8):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        template_feature = self.pooling(self.feature_model(template))
        if maxIteration == 1:
            est_R, est_t, source = self.spam(template_feature, source, est_R, est_t)
        else:
            for i in range(maxIteration):
                est_R, est_t, source = self.spam(template_feature, source, est_R, est_t)
        result = {'est_R': est_R,
                  'est_t': est_t,
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  'transformed_source': source}
        return result

    def spam(self, template_feature, source, est_R, est_t):
        batch_size = source.size(0)
        source_feature = self.feature_model(source)
        self.source_feature = self.pooling(source_feature)
        y = torch.cat([template_feature, self.source_feature], dim=1)
        pose_7d = self.linear(y)
        pose_7d = PCRNetTransform.create_pose_7d(pose_7d)

        # Find current rotation and translation.
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
        est_R_temp = PCRNetTransform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        est_t_temp = PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3)

        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)

        source = PCRNetTransform.quaternion_transform(source, pose_7d)  # Ps' = est_R*Ps + est_t
        return est_R, est_t, source


if __name__ == '__main__':
    template, source = torch.rand(2, 10, 3), torch.rand(2, 10, 3)
    pn = PointNet()
    net = iPCRNet(pn)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
