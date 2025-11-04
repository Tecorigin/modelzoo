#!/bin/bash

# 确保脚本在执行命令的位置出错时立即退出
set -e

# 定义日志文件路径
LOG_FILE="train_sdaa_3rd.log"

# 执行Python命令并将输出重定向到日志文件
python -m paddle.distributed.launch --log_dir=ppyoloe --gpus 2,3 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp > "${LOG_FILE}" 2>&1
