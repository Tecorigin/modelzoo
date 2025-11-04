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

import torch
import torch_sdaa
import torch.backends.cudnn as cudnn

from nets_arcface.arcface import Arcface
from utils.dataloader import LFWDataset
from utils.utils_metrics import test
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--device",
                        default='sdaa', type=str, choices=['cpu', 'cuda', 'sdaa'],
                        help="which device to use, sdaa default")
    parser.add_argument('--model_path', type=str, default="model_data/facenet_mobilenet.pth",
                        help='model_path')
    parser.add_argument('--val_data_path', type=str, default="lfw",
                        help='val_data_path')
    parser.add_argument('--val_pairs_path', type=str, default="model_data/lfw_pair.txt",
                        help='val_data_path')
    args = parser.parse_args()
    #--------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------------#
    if args.device == 'sdaa':
        sdaa         = True
    else:
        sdaa         = False
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone        = "mobilefacenet"
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape     = [112, 112, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    # model_path      = 'model_data/facenet_mobilenet.pth'
    model_path = args.model_path
    # if not os.path.exists(model_path):
    #     print(f"Path does not exist: {model_path}")
    # else:
    #     print(f"Path exists: {model_path}")
    #--------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    lfw_dir_path    = args.val_data_path
    lfw_pairs_path  = args.val_pairs_path
    #--------------------------------------#
    #   评估的批次大小和记录间隔
    #--------------------------------------#
    batch_size      = args.eval_batch_size
    log_interval    = 1
    #--------------------------------------#
    #   ROC图的保存路径
    #--------------------------------------#
    png_save_path   = "model_data/roc_test.png"

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Arcface(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model  = model.eval()

    if sdaa:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        local_rank = 0 
        device = torch.device(f"sdaa:{local_rank}")
        model = model.to(device)

    test(test_loader, model, png_save_path, log_interval, batch_size, sdaa)
