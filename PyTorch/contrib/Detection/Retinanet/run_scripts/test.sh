#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/coco"

#安装依赖
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt

cd $script_path

#执行训练
python run_retinanet.py --config ../configs/retinanet/retinanet_r50_fpn_1x_coco.py\
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options \
        "train_dataloader.dataset.data_root=$data_path" \
        "train_dataloader.dataset.ann_file=$data_path/annotations/instances_train2017.json" \
        "val_dataloader.dataset.data_root=$data_path" \
        "val_dataloader.dataset.ann_file=$data_path/annotations/instances_val2017.json" \
        "val_evaluator.ann_file=$data_path/annotations/instances_val2017.json" 2>&1 | tee sdaa.log

# 生成loss对比图
#python loss.py --sdaa-log sdaa.log --cuda-log cuda.log