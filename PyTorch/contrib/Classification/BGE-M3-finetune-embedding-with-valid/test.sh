#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/bge-m3/finetuning_data/"

#安装依赖

pip install -r requirements.txt

cd $script_path

#执行训练
python train.py 2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log