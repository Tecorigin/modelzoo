#!/bin/bash
# 获取当前脚本的绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 获取项目根目录的路径
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
# 切换到项目根目录
cd "$PROJECT_DIR"
torchrun --nproc_per_node 4 main_plot.py --datapath /data/datasets/cityscapes/ --time_out 7200