#!/bin/bash
cd ..
pip install -r requirements.txt

# torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 train_on_imagenet.py /data/datasets/imagenet/ -a fasternet -b 64 > ./scripts/output.log 2>&1 &
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29502 train_on_imagenet.py /data/datasets/imagenet/ -a fasternet -b 64 > ./scripts/output.log 2>&1 &