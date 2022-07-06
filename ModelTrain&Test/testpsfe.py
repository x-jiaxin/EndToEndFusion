import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40DadaSet import ModelNet40, RegistrationData
# from losses.chamfer_distance import ChamferDistanceLoss
from mse import compute_metrics, summary_metrics
from operations.transform_functions import PCRNetTransform
from trainpsfe import exp_name, MAX_EPOCHS, get_model, dir_name, log_dir

BATCH_SIZE = 4
EVAL = False
START_EPOCH = 0
pretrained = os.path.join(dir_name, 'best_model.t7')  # 使用最好的模型参数测试
print(exp_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def rmse(pts, T, ptt, T_gt):
    pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
    pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
    return torch.norm(pts_pred - ptt, dim=2).mean(dim=1)


def test_one_epoch(device, model, test_loader):
    model.eval()
    count = 0
    test_loss = 0.0
    r_mse, t_mse, r_mae, t_mae = [], [], [], []

    for i, data in enumerate(tqdm(test_loader)):
        template, source, gtT, gtR, gtt = data
        template = template.to(device)
        source = source.to(device)
        gtT = gtT.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        gtt = gtt - torch.mean(source, dim=1).unsqueeze(1)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source)
        # loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        # test_loss += loss_val.item()
        est_R = output['est_R']  # B*3*3
        est_t = output['est_t']  # B*1*3
        gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        cur_r_mse, cur_t_mse, cur_r_mae, cur_t_mae = compute_metrics(est_R, est_t, gtR, gtt)
        r_mse.append(cur_r_mse)
        t_mse.append(cur_t_mse)
        r_mae.append(cur_r_mae)
        t_mae.append(cur_t_mae)
        count += 1
    # test_loss = float(test_loss) / count
    r_mse, t_mse, r_mae, t_mae = summary_metrics(r_mse, t_mse, r_mae, t_mae)
    r_rmse = np.sqrt(r_mse)
    t_rmse = np.sqrt(t_mse)
    # return test_loss, r_rmse, t_rmse, r_mse, t_mse, r_mae, t_mae
    return 0, r_rmse, t_rmse, r_mse, t_mse, r_mae, t_mae


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    testset = RegistrationData(ModelNet40(train=False))
    testloader = DataLoader(testset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            drop_last=False,
                            num_workers=4)
    device = torch.device('cuda:3')
    model = get_model()
    model = model.to(device)
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location='cuda:3'))
    a, b, c, d, e, f, g = \
        test_one_epoch(device, model, testloader)
    info = "Loss:{},\nRMSE(R): {}, RMSE(t): {} & MSE(R): {} & MSE(t): {} & \nMAE(R): {} & MAE(t): {}". \
        format(a, b, c, d, e, f, g)
    print(info)
