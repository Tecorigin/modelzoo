#!/usr/bin/env bash
# 如果你希望在遇到命令报错时“不中断脚本”，
# 建议去掉 set -e 或在命令后面加 "|| true"。
# 这里示例不使用 set -e，以便即使 timeout 退出也能继续下一个模型。
# 每条命令限制6小时运行时间
cd ..
pip install -r requirements.txt
cd examples/imagenet
torchrun --nproc_per_node=4 mmain.py --arch efficientnet-b6 -b 4