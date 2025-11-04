#!/bin/bash
pip install -r requirements.txt

python train.py --cuda -d coco --root /mnt/nvme1/application/zhaoyt/dataset -m yolov1 -bs 4 --max_epoch 1 --wp_epoch 1 --eval_epoch 10 --fp16 --ema --multi_scale > yolov1.log 2>&1 &

cd scripts
python loss_check.py