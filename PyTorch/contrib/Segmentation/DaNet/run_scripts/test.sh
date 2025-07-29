#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/Cityscapes/"

#安装依赖
# cd .. 
# pip3 install -U openmim
# pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
# pip install opencv-python==4.7.0.72
# pip install "mmcv==2.1.0"
# pip install -e .
# pip install -r requirements.txt
# pip3 install numpy==1.24.3
# pip install git+https://gitee.com/xiwei777/tcap_dllogger.git


cd $script_path

#执行训练
python run_danet.py --config ../configs/danet/danet_r50-d8_4xb2-80k_cityscapes-512x1024.py \
    --launcher pytorch --nproc-per-node 4 --amp\
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log

#绘制loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log