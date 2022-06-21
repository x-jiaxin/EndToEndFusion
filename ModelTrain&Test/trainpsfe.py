import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40DadaSet import ModelNet40, RegistrationData
from losses.chamfer_distance import ChamferDistanceLoss
# from losses.frobenius_norm import FrobeniusNormLoss
from model.PSFENet import PSFENet
from operations.transform_functions import PCRNetTransform
from utils.SaveLog import SaveLog
from visdom import Visdom

BATCH_SIZE = 32
START_EPOCH = 0
MAX_EPOCHS = 200
gpu_ids = [0, 1, 2]
device = torch.device("cuda:0")
pretrained = ""  # 是否有训练过的模型可用s
resume = ""  # 最新的检查点文件

exp_name = "PSFENet_CDLoss_128"

dir_name = os.path.join(
    os.path.dirname(__file__), os.pardir, "checkpoints4", exp_name, "models"
)
log_dir = os.path.join(os.path.dirname(__file__), 'log')

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_model():
    return PSFENet()


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt, gtR, gtt = data
        template = template.to(device)  # [B,N,3]
        source = source.to(device)  # [B,N,3]
        igt = igt.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)
        # gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        output = model(template, source)
        # loss_val = FrobeniusNormLoss()(output["est_T"], gtT)
        loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        train_loss += loss_val.item()
        # 批次加一
        count += 1
    train_loss = float(train_loss) / count
    return train_loss


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, gtR, gtt = data
        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)
        # gtt = gtt - torch.mean(source, dim=1).unsqueeze(1)  # ?
        # gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        output = model(template, source)
        # loss_val = FrobeniusNormLoss()(output["est_T"], gtT)
        loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        test_loss += loss_val.item()
        count += 1
    test_loss = float(test_loss) / count
    return test_loss


def train(model, train_loader, test_loader):
    vis = Visdom(env='main')
    opts = dict(
        xlabel="Epoch",
        ylabel="Loss",
        title=f"{exp_name}",
        legend=["train", "test"])
    startTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(learnable_params)
    # optimizer = torch.optim.Adam(learnable_params, lr=0.0001)

    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_test_loss = np.inf
    # 绘制起点
    vis.line(X=[0, ], Y=[[0., 0.]], opts=opts, win='train')
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(device, model, train_loader, optimizer)
        test_loss = test_one_epoch(device, model, test_loader)
        # 可视化loss
        vis.line([[train_loss, test_loss]], [epoch + 1], win='train', update='append')
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                snap, os.path.join(dir_name, "best_model_snap.t7"))
            torch.save(
                model.module.state_dict(),
                os.path.join(dir_name, "best_model.t7"),
            )
        torch.save(
            snap,
            os.path.join(dir_name, "model_snap.t7")
        )
        torch.save(
            model.module.state_dict(),
            os.path.join(dir_name, "model.t7")
        )

        info = "EPOCH:{},Training Loss:{},Testing Loss:{},Best Loss:{}". \
            format(epoch + 1, train_loss, test_loss, best_test_loss)
        print(info)
        IO = SaveLog(os.path.join(log_dir, f'{exp_name}.log'))
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        IO.savelog(current_time + "\n" + info)
    endTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(exp_name)
    print(f"Over!\nStart time:{startTime}\nEnd time:{endTime}")


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    trainset = RegistrationData(ModelNet40(train=True))
    testset = RegistrationData(ModelNet40(train=False))
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4
    )
    model = get_model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume)
        START_EPOCH = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))
    train(model, trainloader, testloader)
