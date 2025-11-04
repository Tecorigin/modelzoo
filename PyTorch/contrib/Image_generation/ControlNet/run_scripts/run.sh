#!/bin/bash

BASE_DIR=$(cd "$(dirname "$0")/.." && pwd)  # 自动获取当前脚本上层目录（ControlNet 根目录）

MODEL_NAME="controlnet"
BATCHSIZE=1
EPOCH=1
MAX_ITER=100
LR=1e-5
CKPT="${BASE_DIR}/models/control_sd15_ini.ckpt"
LOGFILE="sdaa.log"

echo "[INFO] BASE_DIR is ${BASE_DIR}"
echo "[INFO] Starting training..."
python ${BASE_DIR}/run_scripts/run_controlnet.py \
    --model_name ${MODEL_NAME} \
    --batchsize ${BATCHSIZE} \
    --epoch ${EPOCH} \
    --max_iter ${MAX_ITER} \
    --lr ${LR} \
    --ckpt ${CKPT} \
    --logfile ${LOGFILE}
