#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/Cityscapes"

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
python run_bisenetv1.py --config ../configs/bisenetv1/bisenetv1_r18-d32_4xb4-160k_cityscapes-1024x1024.py \
       --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
