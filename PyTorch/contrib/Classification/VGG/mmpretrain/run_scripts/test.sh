#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 数据集路径
data_path="/data/teco-data/imagenet"

# 安装依赖
echo "正在安装Python依赖..."
cd .. 
pip3 install  -U openmim 
pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
pip3 install opencv_python mmcv --no-deps
mim install -e .
pip install -r requirements.txt
pip3 install numpy==1.24.3

cd $script_path/

# 启动训练
echo "开始训练..."
torchrun --master_port=29600 run_vgg.py \
    --config ../configs/vgg/vgg11_8xb32_in1k.py \
    --launcher pytorch \
    --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" \
    "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log

# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log

echo "训练完成！结果保存在:"
echo " - 训练日志: $script_path/sdaa.log"
echo " - Loss曲线: $script_path/loss.jpg"