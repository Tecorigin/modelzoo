#!/bin/bash

# 示例运行命令，替换为你实际需要的参数
python ../src/open_clip_train/main.py \
  --batch-size 32 \
  --epochs 10 \
  --learning-rate 0.001 \
  --model "ViT-B-32" \
  --data-path "/data/dataset" \
  --output-dir "./output" \
  --use-amp
