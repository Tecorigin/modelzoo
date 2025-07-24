#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"
data_path="/data/teco-data/VOC/VOCdevkit/VOC2007"
# data_path="/data02/VOC/VOCdevkit/VOC2007"

#安装依赖
cd ..
pip install -U pip
pip install setuptools==80.9.0
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install git+https://gitee.com/xiwei777/tcap_dllogger.git
pip3 install mmcv==2.1.0
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3

cd $script_path

#执行训练
python run_csra.py --config ../configs/csra/resnet101-csra_1xb16_voc07-448px.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" "randomness.seed=42" "randomness.deterministic=True" 2>&1 | tee sdaa.log

# 生成loss对比图
# python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
