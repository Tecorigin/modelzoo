#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt

# sdaa 上的训练脚本
python run_resnext50_32_4d.py \
--dataset_path /data/teco-data/imagenet \
--batch_size 32 \
--epochs 1 \
--lr 0.01 \
--amp \
--save_path ../checkpoints \
--max_step 100 \
--device sdaa \
2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
