#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

#安装依赖
cd .. 
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numpy==1.24.3
pip install huggingface_hub
pip install parameterized
cd $script_path
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
cd ..


#执行训练

python run_transformerxl.py 2>&1 | tee sdaa.log


# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
