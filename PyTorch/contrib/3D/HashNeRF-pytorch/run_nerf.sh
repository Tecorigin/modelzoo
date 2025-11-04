#!/bin/bash
# run_nerf.sh

# 设置环境变量
#export TORCH_SDAA_LOG_LEVEL="debug"  # 替换为实际需要的环境变量
#export SDAA_LAUNCH_BLOCKING=1

# 如果你需要设置Python路径或其他环境配置，也可以在这里做

# 执行Python脚本，并传递命令行参数
python run_nerf.py \
    --config configs/hotdog.txt \
    --finest_res 512 \
    --log2_hashmap_size 19 \
    --lrate 0.01 \
    --lrate_decay 10