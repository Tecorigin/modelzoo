import argparse
import random
from copy import deepcopy

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import os
import torch
import torchvision
from pathlib import Path
import os
import glob
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from functools import partial
import numpy as np
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from image_classification.models import resnet50
from image_classification.utils import *
# from image_classification.dataloaders import *

def fast_collate(memory_format, batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array.copy())

    return tensor, targets

def get_pytorch_val_loader(
    data_path,
    image_size,
    batch_size,
    interpolation="bilinear",
    workers=5,
    crop_padding=32,
    memory_format=torch.contiguous_format,
    prefetch_factor=2,
    rank=-1,
):
    interpolation = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}[
        interpolation
    ]
    valdir = os.path.join(data_path, "val")
    
    transforms_list = [
                transforms.Resize(image_size + crop_padding, interpolation=interpolation),
                transforms.CenterCrop(image_size),
    ]
    
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose(transforms_list),
    )

    if torch.distributed.is_initialized():
        
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        num_workers=workers,
        worker_init_fn=None,
        pin_memory= True ,
        collate_fn=partial(fast_collate, torch.contiguous_format),
        drop_last=True ,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    return (val_loader,len(val_loader))


def str2bool(v):
    """
    将命令行输入的str转换为布尔值
    :param v: str值
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/data/application/common/imagenet', help='images path')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--target', default='cuda', help='sdaa or cpu or cuda')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    
    torch.manual_seed(1234)
    np.random.seed(seed=1234)
    random.seed(1234)
        
    acc_list = []
    
    device= opt.target
    if opt.target=='cuda':
        from torch.cuda.amp import autocast
    else:
        from torch_sdaa.amp import autocast
    
    get_val_loader = get_pytorch_val_loader

    val_loader, val_loader_len = get_val_loader(
                data_path=opt.data_path,
                image_size=224,
                batch_size=opt.batch_size,
            )
    data_iter = enumerate(val_loader)

    model = resnet50(pretrained=True).to(device)
    
    
    for i,(input,target) in data_iter:
        if True:
            input,target = input.to(device),target.to(device)
            input = input.float()
            mean = (
                        torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).to(f'{device}')
                        .view(1, 3, 1, 1)
                    )
            std = (
                        torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).to(f'{device}')
                        .view(1, 3, 1, 1)
                    )
            input = input.sub_(mean).div_(std)
            
        bs = input.size(0)
        
        input = input.to(memory_format=torch.channels_last)
        
        with torch.no_grad(), autocast(enabled=True):
            output = model(input)
        
        with torch.no_grad():
            precs = accuracy(output.data, target, topk=(1, 1))
        precs = map(lambda t: t.item(), precs)
        infer_result = {f"top{k}": (p, bs) for k, p in zip((1, 1), precs)}
        
        # top1.record(infer_result["top1"][0], bs)
        
        
        
        acc_list.append(infer_result["top1"][0])
        
        print('acc:',sum(acc_list)/len(acc_list))

    
    print('eval_accuracy:',sum(acc_list)/len(acc_list))
        
        
