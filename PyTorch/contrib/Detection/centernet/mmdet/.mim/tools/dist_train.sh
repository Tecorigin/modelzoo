#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}  # 根据你的需求设置合适的值
MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}  # 根据你的需求设置合适的值
export PYTHONWARNINGS="ignore::UserWarning"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
