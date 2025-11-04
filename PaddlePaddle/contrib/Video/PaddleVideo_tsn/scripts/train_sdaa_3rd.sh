#!/bin/bash
cd ..
# 确保脚本在执行命令的位置出错时立即退出
set -e

# 定义日志文件路径
LOG_FILE="script/train_sdaa_3rd.log"
export PADDLE_XCCL_BACKEND=sdaa
# 执行Python命令并将输出重定向到日志文件

python -B -m paddle.distributed.launch --gpus="0,1,2,3" --log_dir=log_tsn main.py  --validate -c configs/recognition/tsn/tsn_k400_frames.yaml > "${LOG_FILE}" 2>&1