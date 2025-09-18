#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path = "/data/teco-data/coco"
train_path="$data_path/train2017"
val_path="$data_path/val2017"

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3
pip install "pycocoevalcap"

cd $script_path

#执行训练
python run_blip.py --config ../configs/blip/blip-base_8xb32_caption.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$train_path" "val_dataloader.dataset.data_root=$val_path" 2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log