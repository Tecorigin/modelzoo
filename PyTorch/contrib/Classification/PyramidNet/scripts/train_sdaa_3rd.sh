#!/bin/bash

pip install -r ../requirements.txt

# 获取系统的总 CPU 核心数
TOTAL_CORES=$(nproc --all)

# 设置 PyTorch 使用的 GPU 数
NUM_PROCESSES=4

# 计算每个进程应该使用的线程数
OMP_NUM_THREADS=$((TOTAL_CORES / NUM_PROCESSES))

# 确保 OMP_NUM_THREADS 至少为 1
if [ "$OMP_NUM_THREADS" -lt 1 ]; then
    OMP_NUM_THREADS=1
fi

# 输出设定值
echo "Total CPU Cores: $TOTAL_CORES"
echo "Number of Training Processes: $NUM_PROCESSES"
echo "Setting OMP_NUM_THREADS=$OMP_NUM_THREADS"

# 设置环境变量
export OMP_NUM_THREADS=$OMP_NUM_THREADS

# 设置数据路径和权重路径
DATASET_PATH="/data/datasets/imagenet"

# 使用torchrun启动分布式训练，4个进程
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_PROCESSES ../train.py \
    --dataset imagenet \
    --data ${DATASET_PATH} \
    --net_type pyramidnet \
    --depth 164  \
    --alpha 48 \
    --batch_size 64 \
    --lr 0.5 \
    --epochs 1 \
    --expname PyramidNet-200 \
    --print-freq 1 | tee train_sdaa_3rd.log