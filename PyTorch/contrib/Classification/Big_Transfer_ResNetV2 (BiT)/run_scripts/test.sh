#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt

# sdaa 上的训练脚本
python3 -m bit_pytorch.train \
--name imagenet_`date +%F_%H%M%S` \
--model BiT-M-R50x1 \
--logdir /tmp/bit_logs \
--dataset imagenet2012 \
--datadir /data/teco-data/imagenet \
--batch 4 \
--base_lr 0.00025 \
--workers 8 \
--eval_every 100 \
--batch_split 4 \
2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log