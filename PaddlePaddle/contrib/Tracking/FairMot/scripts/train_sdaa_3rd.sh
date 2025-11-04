#!/bin/bash
cd ..
# 确保脚本在执行命令的位置出错时立即退出
set -e

# 定义日志文件路径
LOG_FILE="scripts/train_sdaa_3rd.log"
export PADDLE_XCCL_BACKEND=sdaa
# 执行Python命令并将输出重定向到日志文件
python -m paddle.distributed.launch --log_dir=./fairmot_dla34_30e_1088x608/ --gpus 0,1,2,3 tools/train.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608_airplane.yml > "${LOG_FILE}" 2>&1
