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
pip install huggingface_hub
pip install parameterized
pip install transformers
cd $script_path

#执行训练
#!/bin/bash

# 训练参数配置

python run_electra.py --do_train --do_eval 2>&1 | tee sdaa.log

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
