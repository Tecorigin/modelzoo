# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 
import matplotlib.pyplot as plt

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def plot_train_loss(train_losses, work_dir):
    plt.figure()
    # 绘制训练损失曲线
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train Loss Curve')
    train_loss_path = os.path.join(work_dir, 'train_loss_curve.jpg')
    plt.savefig(train_loss_path)
    plt.close()

def plot_val_loss(val_losses, epoch_numbers, work_dir):
    plt.figure()
    # 绘制验证损失曲线
    plt.plot(epoch_numbers, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Validation Loss Curve')
    val_loss_path = os.path.join(work_dir, 'val_loss_curve.jpg')
    plt.savefig(val_loss_path)
    plt.close()

def get_new_experiment_folder(base_path="experiments"):
    """
    检查已有的实验文件夹数量，并生成新的实验文件夹名
    如果新文件夹已存在，则生成新的实验文件夹，编号为当前experiments中存在的实验文件夹的最大编号+1
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("exp")]
    existing_numbers = [int(d[3:]) for d in existing_folders if d[3:].isdigit()]
    new_exp_number = max(existing_numbers, default=0) + 1
    
    while True:
        new_exp_folder = os.path.join(base_path, f"exp{new_exp_number}")
        if not os.path.exists(new_exp_folder):
            os.makedirs(new_exp_folder)
            break
        new_exp_number += 1

    return new_exp_folder