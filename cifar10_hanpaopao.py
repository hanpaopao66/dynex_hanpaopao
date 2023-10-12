#!/usr/local/bin python3.8
# -*- coding: utf-8 -*-
# @Time : 2023/10/12 6:16 PM

import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入 Dynex Pytorch 图层类
# # Dynex 平台可用作神经形态 PyTorch 层。 我们导入所需的类：
from HybridQRBM.pytorchdnx import dnx
from HybridQRBM.optimizers import RBMOptimizer
from HybridQRBM.samplers import DynexSampler

# 我们定义训练超参数。 神经形态 Dynex 层以极快的速度进化到接近最佳的基态。 因此，我们只需要几个训练周期就可以得到完全训练的模型。INIT_LR = 1e-3
BATCH_SIZE = 10000  # number of images per batch
EPOCHS = 2  # number of training epochs
device = "cpu"  # no GPU needed, we compute on the Dynex platform

optimizer = RBMOptimizer(
    learning_rate=0.05,
    momentum=0.9,
    decay_factor=1.00005,
    regularizers=()
)

sampler = DynexSampler(mainnet=True,
                       num_reads=10000,
                       annealing_time=200,
                       debugging=False,
                       logging=True,
                       num_gibbs_updates=1,
                       minimum_stepsize=0.002)


class QModel(nn.Module):
    def __init__(self, n_hidden, steps_per_epoch, sampler, optimizer):
        super().__init__()
        # Dynex 神经形态层
        self.dnxlayer = dnx(n_hidden, steps_per_epoch, sampler=sampler,
                            optimizer=optimizer)

    def forward(self, x):
        x = self.dnxlayer(x)
        return x


from torchvision import transforms


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)


data_transformer = transforms.Compose([
    transforms.ToTensor(),
    ThresholdTransform(thr_255=240)
])

# 加载 CIFAR 10 数据集
from torchvision.datasets import CIFAR10

trainData = CIFAR10(root="data", train=True, download=True,
                    transform=data_transformer)
testData = CIFAR10(root="data", train=False, download=True,
                   transform=data_transformer)
print("[INFO] CIFAR dataset lodaed")

# 初始化训练、验证和测试数据加载器
trainDataLoader = DataLoader(trainData, shuffle=True, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
model = QModel(60, trainSteps, sampler, optimizer)

import os

for e in range(1, EPOCHS + 1):
    print('EPOCH', e, 'of', EPOCHS)
    # 将模型设置为训练模式
    model.train()
    print(model.state_dict())
    correct = 0
    total = 0
    train_loss = 0

    best_acc = 0

    # 循环训练集
    for (x, y) in trainDataLoader:
        # 将输入发送到设备 device
        (x, y) = (x.to(device), y.to(device))
        # 执行前向传播并计算训练损失
        pred = model(x)
        acc = model.dnxlayer.acc[-1]
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': e,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model, './checkpoint/cifat10_pp_dynex.pth')
            best_acc = acc

print('FOUND MODEL ACCURACY:', np.array(model.dnxlayer.acc).max(), '%')
