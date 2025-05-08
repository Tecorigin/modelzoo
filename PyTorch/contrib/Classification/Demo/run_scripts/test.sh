#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt
# 数据集路径,保持默认
data_path="/data/imagnet"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

python run_demo.py --nproc_per_node 4 --model_name demo --epoch 1 --batch_size 32 --device sdaa --step 100 --datasets $dataset 2>&1 | tee sdaa.log

python loss.py --sdaa-log sdaa.log --cuda-log cuda.log