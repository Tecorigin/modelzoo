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
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms


def process(img, resize_shape=256, crop_shape=224):
    img_transforms = transforms.Compose(
        [transforms.Resize(resize_shape), transforms.CenterCrop(crop_shape), transforms.ToTensor()]
    )
    img = img_transforms(img)

    with torch.no_grad():
        # mean and std are not multiplied by 255 as they are in training script
        # torch dataloader reads data into bytes whereas loading directly
        # through PIL creates a tensor with floats in [0,1] range
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = img.float()

        input = img.unsqueeze(0).sub_(mean).div_(std)
    input = input.numpy()
    return input


def preprocess(image_path, dtype='float16', resize_shape=256, crop_shape=224):
    images = []
    if isinstance(image_path, str):
        if os.path.isfile(image_path):
            img = process(Image.open(image_path), resize_shape, crop_shape)
            images = [img]
        else:
            print("无法打开图片文件:", image_path)
            return None
    elif isinstance(image_path, Image.Image): #判断 Image 类型
        img = process(image_path, resize_shape, crop_shape)
        images = [img]
    elif isinstance(image_path[0],str): #判断 [str] 类型
        for i in image_path:
            img = process(Image.open(image_path), resize_shape, crop_shape)
            images.append(img)
    elif isinstance(image_path[0],Image.Image): #判断 [Image] 类型
        for i in image_path:
            img = process(i, resize_shape, crop_shape)
            images.append(img)
    else:
        print("输入有误")
        return None
    
    images = np.vstack(images)
    images = images.astype(np.float16) if dtype=='float16' else images.astype(np.float32)
    return images