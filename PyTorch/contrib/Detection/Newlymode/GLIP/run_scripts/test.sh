#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

data_path="/data/teco-data/coco/"

#安装依赖
#运行前请取消注释
cd .. 
# pip3 install  -U openmim 
# pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
# pip3 install opencv_python mmcv --no-deps
# mim install -e .
# pip install -r requirements.txt

cd $script_path

#执行训练
python run_GLIP.py --config ../configs/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py \
    --launcher pytorch --nproc-per-node 1 2>&1 | tee sdaa.log  \
    # --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path"  # 不注释掉会额外传参导致报错

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log