#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 安装依赖
echo "正在安装Python依赖..."
### 安装环境的 交上去的时候取消注释 下面三行
# cd $script_path/../
# pip install -r requirements.txt
# python setup.py develop --no_cuda_ext

# # 数据集路径设置
# data_path="$script_path/../datasets/SIDD"
# train_zip="/path/to/train/SIDD-train.zip"
# val_zip="/path/to/val/SIDD-val.zip"

# # 准备数据集
# echo "准备数据集..."
# mkdir -p $data_path/Data
# unzip $train_zip -d $data_path/Data
# unzip $val_zip -d $data_path

# echo "处理训练数据..."
# python $script_path/../scripts/data_preparation/sidd.py

# 训练参数配置
config_file="$script_path/cae/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k.py"
log_file="$script_path/work_dirs/cae_beit-base-p16_8xb256-amp-coslr-300e_in1k/logs/sdaa.log"
cuda_log_file="$script_path/cuda.log"
train_file="$script_path/run_cae.py"

# 启动训练
echo "开始训练..."
cd $script_path/
#跑的命令
torchrun  \
      --master_port=29500 \
      $train_file \
      $config_file \
      --launcher pytorch


# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py \
    --sdaa-log $log_file \
    --cuda-log $cuda_log_file

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"