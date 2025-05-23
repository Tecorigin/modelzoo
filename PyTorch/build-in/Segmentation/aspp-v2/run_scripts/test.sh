#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt

# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

# 使用的torchsdaa野包需要导入.so文件
# export LD_LIBRARY_PATH=/root/miniconda3/envs/torch_env_py310/lib/python3.10/site-packages/torch_sdaa/lib:$LD_LIBRARY_PATH

#如长训请提供完整命令即可，100iter对齐提供100iter命令即可
python run_ASPP_V2.py --model_name ASPPV2 \
        --data_path /mnt_qne00/dataset/ \
        --device sdaa \
        --num_classes 20 \
        --batch_size 4 \
        --epochs 3 \
        --lr 0.0001 \
        | tee sdaa.log
#生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log