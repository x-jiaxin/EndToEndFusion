"""
@Date: 2022/06/20 10:53
"""
import torch


def knn(x, k=20):
    x = x.transpose(1, 2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    idx = distance.topk(k=k, dim=-1)[1]
    return idx


def get_local_feature(x, refer_idx):
    x = x.transpose(1, 2).contiguous()  # BNC
    x = x.view(*x.size()[:3])

    batch_size, num_points, k = refer_idx.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = refer_idx + idx_base

    idx = idx.view(-1)

    _, _, num_dims = x.size()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k * num_dims)

    feature = feature.transpose(1, 2).contiguous()  # B,kC,N
    return feature


class PSFE(torch.nn.Module):
    def __init__(self, k=20, input_shape='bnc', pt_dim=64, out_dim=256):
        super().__init__()
        self.k = k
        self.pt_dim = pt_dim
        self.input_shape = input_shape
        self.out_dim = out_dim
        self.conv1 = torch.nn.Conv1d(3, self.pt_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.pt_dim, self.pt_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.pt_dim, self.pt_dim, 1)
        self.conv4 = torch.nn.Conv1d(self.pt_dim, 128, 1)

        self.conv1d_L1 = torch.nn.Conv1d(3 * self.k, self.out_dim, 1)
        self.conv1d_L2 = torch.nn.Conv1d(self.pt_dim * self.k, self.out_dim, 1)
        self.conv1d_L3 = torch.nn.Conv1d(self.pt_dim * self.k, self.out_dim, 1)
        self.conv1d_L4 = torch.nn.Conv1d(self.pt_dim * self.k, self.out_dim, 1)
        self.conv1d_L5 = torch.nn.Conv1d(128 * self.k, self.out_dim, 1)

    def forward(self, x):
        if self.input_shape == 'bnc':
            x = x.transpose(1, 2)
        idx = knn(x, self.k)

        x = self.conv1(x)
        x1 = get_local_feature(x, idx)
        x1 = self.conv1d_L2(x1)

        x = self.conv2(x)
        x2 = get_local_feature(x, idx)
        x2 = self.conv1d_L3(x2)

        x = self.conv3(x)
        x3 = get_local_feature(x, idx)
        x3 = self.conv1d_L4(x3)

        x = self.conv4(x)
        x4 = get_local_feature(x, idx)
        x4 = self.conv1d_L5(x4)

        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out


if __name__ == '__main__':
    a = torch.rand(2, 100, 3).cuda()
    model = PSFE(k=20, input_shape='bnc').cuda()
    out = model(a)
    print(out.shape)
