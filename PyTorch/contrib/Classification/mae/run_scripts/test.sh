#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/imagenet"

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3

cd $script_path

#执行训练
python run_mae.py --config ../configs/mae/mae_hivit-base-p16_8xb512-amp-coslr-400e_in1k.py \
    --launcher pytorch --nproc-per-node 1 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path"  2>&1 | tee sdaa.log
# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
