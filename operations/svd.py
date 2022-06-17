import torch
import torch.nn as nn


class SVD(nn.Module):
    def __init__(self, emb_dims, input_shape="bnc"):
        super(SVD, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.input_shape = input_shape

    def forward(self, source, template, pose):
        batch_size = source.size(0)
        if self.input_shape == "bnc":
            source = source.permute(0, 2, 1)  # [B,3,N]
            template = template.permute(0, 2, 1)

        # 经过一个pose变换后的源点云Pt'
        tentative_transform = source + pose

        source_centered = source - source.mean(dim=2, keepdim=True)
        tentative_transform_centered = tentative_transform - tentative_transform.mean(dim=2, keepdim=True)

        H = torch.matmul(source_centered, tentative_transform_centered.permute(0, 2, 1).contiguous())
        U, S, V = [], [], []
        R = []

        for i in range(source.size(0)):
            u, s, v = torch.svd(H[i].data)
            # u, s, v = torch.linalg.svd(H[i])
            u = u.to(source.device)
            s = s.to(source.device)
            v = v.to(source.device)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            reflect = self.reflect
            reflect = reflect.to(source.device)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                r = r * reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)
        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, source.mean(dim=2, keepdim=True)) + tentative_transform.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)
