#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo "当前脚本路径: $script_path"

#安装依赖
cd .. 
pip install -r requirements.txt
pip3 install numpy==1.24.3
pip install huggingface_hub
pip install parameterized

cd $script_path

#执行训练

python run_electra.py   \
--model_name_or_path ../configs/electra-small-generator   \
--train_file ../configs/train.txt   \
--do_train --do_eval   \
--output_dir electra_out   \
--overwrite_output_dir   \
--per_device_train_batch_size 16   \
--max_seq_length 32   \
--line_by_line

# 生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log
