#!/bin/bash

pip install -r ../requirements.txt
pip install torchvision==0.15.2 --no-deps

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

# 启动 PyTorch 训练
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_PROCESSES ../tools/train.py -c ../configs/rtdetr/rtdetr_r50vd_6x_coco.yml  2>&1 | tee train_sdaa_3rd.log