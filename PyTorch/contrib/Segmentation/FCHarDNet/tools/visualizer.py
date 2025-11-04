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

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
from argparse import ArgumentTypeError
import numpy as np
from PIL import Image
import cv2

import torch
import torch_sdaa
import torch.nn as nn

from torchvision import transforms
import torch.nn.functional as F

from models import FCHarDNet
from utils import validate, get_dataset, plot_train_loss, plot_val_loss, get_new_experiment_folder

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--ckpt", default='experiments/example/best_fchardnet_vaihingen.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--img_file", type=str, default='datasets/data/vaihingen_VOC/VOC2007/JPEGImages/area1_0_0_512_512.png',
                        help="path to Dataset")
    parser.add_argument("--alpha", type=int, default=0.5,
                        help="transparency of mask")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="num classes (default: None)")
    parser.add_argument("--model_name", type=str, default='fchardnet',
                        choices=['fchardnet', ], help='model name')
    parser.add_argument("--mix", type=str2bool, default=True,
                        help="mix segmentation result with original image")
    parser.add_argument('--device', required=True, default='sdaa', type=str,
                        help='which device to use. cuda, sdaa optional, sdaa default')

    return parser


def main():
    opts = get_argparser().parse_args()
    
    # 定义类别及其对应的颜色
    CLASS_NAMES = ['Impervious surfaces', 'Building', 'Low vegetation', 'Tree', 'Car', 'Clutter']
    CLASS_COLORS = [
        (255, 255, 255),
        (0, 0, 255),   
        (0, 255, 255),   
        (0, 255, 0), 
        (255, 255, 0), 
        (255, 0, 0) 
    ]


    if opts.ckpt == None or not os.path.isfile(opts.ckpt):
        print(f"Checkpoint directory {opts.ckpt} does not exist.")
        return

    vis_path = os.path.join(os.path.dirname(opts.ckpt), 'vis_results')

    device = torch.device(opts.device if torch.sdaa.is_available() else 'cpu')
    print(f'use device:[{device}]')

    if opts.model_name == "fchardnet":
        model = FCHarDNet(opts.num_classes).to(device)
        print('model is loaded successfully')
    else:
        print('model_name error')
        return
    
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    print('checkpoint file loaded successfully')
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_image = Image.open(opts.img_file).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_batch)
        output = F.softmax(output, dim=1)
        output = output.argmax(dim=1).squeeze().cpu().numpy()
        print(output.shape)

    image = cv2.imread(opts.img_file)

    print(opts.mix)

    if opts.mix:
        overlay = image.copy()
        for class_id, color in enumerate(CLASS_COLORS):
            overlay[output == class_id] = color
        cv2.addWeighted(overlay, opts.alpha, image, opts.alpha, 0, image)
    else:
        colored_mask = np.zeros_like(image)
        for class_id, color in enumerate(CLASS_COLORS):
            colored_mask[output == class_id] = color
        image = colored_mask
    
    # 保存可视化结果
    os.makedirs(vis_path, exist_ok=True)
    vis_result_path = os.path.join(vis_path, f'vis_{os.path.basename(opts.img_file)}')
    cv2.imwrite(vis_result_path, image)
    print(f"Saved visualization to {vis_result_path}")


if __name__ == "__main__":
    main()
