#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

# 添加 PYTHONPATH，确保能找到 mmpretrain 模块
# 自动推导 mmpretrain 模块所在路径
export PYTHONPATH=$(realpath "$script_path/../mmpretrain"):$PYTHONPATH

# 安装依赖
echo "正在安装Python依赖..."
### 安装环境的 交上去的时候取消注释 下面三行
# cd $script_path/../
# pip install -r requirements.txt
# python setup.py develop --no_cuda_ext

# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/imagenet"

# 训练参数配置
config_file="$script_path/swin_transformer/swin-tiny_16xb64_in1k.py"
log_file="$script_path/sdaa.log"
cuda_log_file="$script_path/cuda.log"
train_file="$script_path/run_swin_transformer.py"

# 启动训练
echo "开始训练..."
cd $script_path/
#跑的命令
torchrun  \
      --master_port=29500 \
      $train_file \
      $config_file \
      --launcher pytorch \
      --amp | tee sdaa.log

# 生成loss曲线图
echo "生成训练结果图表..."
python loss.py \
    --sdaa-log $log_file \
    --cuda-log $cuda_log_file

echo "训练完成！结果保存在:"
echo " - 训练日志: $log_file"
echo " - Loss曲线: $script_path/loss.jpg"