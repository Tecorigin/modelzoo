#!/bin/bash
cd ..
# 确保脚本在执行命令的位置出错时立即退出
set -e

# 定义日志文件路径
LOG_FILE="scripts/train_sdaa_3rd.log"
# 执行Python命令并将输出重定向到日志文件
torchrun --nproc_per_node 4 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml > "${LOG_FILE}" 2>&1
