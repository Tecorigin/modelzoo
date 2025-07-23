# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reuse the above copyright notice,
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
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import argparse
from ultralytics import YOLO
import ruamel.yaml

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    # /data/teco-data/COCO
    parser = argparse.ArgumentParser(description='YOLOv12 training script for SDAA')
    parser.add_argument('--model_name', required=True, default='yolov12', type=str, help='name of the model')
    parser.add_argument('--total_epochs', type=int, default=600, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32*8, type=int, help='Input batch size for all devices')
    parser.add_argument('--device', type=int, default=8, help='number of devices')
    parser.add_argument('--data_path', required=True, type=str, help='path to dataset configuration (e.g., coco.yaml)')
    parser.add_argument('--early_stop', type=int, default=-1, help='early stop training epochs')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--autocast', default=True, type=str2bool, help='open autocast for amp')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='warmup initial bias lr')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam', 'RMSProp', 'auto'], default='SGD', help='optimizer')
    parser.add_argument('--imgsz', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale augmentation')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup augmentation probability')
    parser.add_argument('--copy_paste', type=float, default=0.1, help='copy-paste augmentation probability')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    try:
        import torch_sdaa
        from torch_sdaa.utils import cuda_migrate
    except ImportError:
        print("Error: torch_sdaa is required for SDAA support. Please install it first.")
        exit(1)

    # Create YAML parser
    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True  # Preserve original quotes
    # Read YAML file
    with open('ultralytics/cfg/datasets/coco.yaml', 'r') as file:
        config = yaml.load(file)

    # Modify path field
    config['path'] = args.data_path

    # Write back to YAML file
    with open('ultralytics/cfg/datasets/coco.yaml', 'w') as file:
        yaml.dump(config, file)

    print(f"Data path updated to: {args.data_path}")

    # Initialize YOLO model
    model = YOLO(f'ultralytics/cfg/models/12/{args.model_name}.yaml')

    # Train the model on SDAA
    results = model.train(
        data="coco.yaml",
        epochs=args.total_epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        scale=args.scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        device=[i for i in range(int(args.device)*4)],
        workers=args.num_workers,
        lr0=args.lr0,
        warmup_bias_lr=args.warmup_bias_lr,
        optimizer=args.optimizer,
        auto_augment=args.autocast,
        patience=args.early_stop if args.early_stop > 0 else None
    )