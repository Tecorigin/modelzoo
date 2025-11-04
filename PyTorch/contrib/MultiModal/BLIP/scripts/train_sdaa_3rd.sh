#!/bin/bash
cd ..
#LOG_FILE="scripts/train_sdaa_3rd.log"
# 执行Python命令并将输出重定向到日志文件
export HF_ENDPOINT="https://hf-mirror.com"
python -m torch.distributed.run --nproc_per_node=4 train_caption.py 