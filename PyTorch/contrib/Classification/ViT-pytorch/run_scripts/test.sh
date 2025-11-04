#!/bin/bash
script_path=$(dirname $(readlink -f "$0"))
echo $script_path

#安装依赖
pip3 install -r ../requirements.txt
cd ../
cd checkpoint
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
cd../
cd $script_path
# 数据集路径,保持默认统一根目录即可
data_path="/data/teco-data/cifar10"
# # 参数校验
# for para in $*
# do
#     if [[ $para == --data_path* ]];then
#         data_path=`echo ${para#*=}`
# done

#如长训请提供完整命令即可，100iter对齐提供100iter命令即可

export TORCH_SDAA_AUTOLOAD=cuda_migrate

#示例1: python run_resnet.py --nproc_per_node 4 --model_name resnet50 --epoch 1 --batch_size 32 --device sdaa --step 100 --datasets $dataset 2>&1 | tee sdaa.log
#由于demo无需下载数据集及数据集太小所以未做step适配，正常场景参考示例1即可
python run_vit.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --train_batch_size 64 --fp16_opt_level 01 --num_steps 100 2>&1 | tee sdaa.log

#生成loss对比图
python loss.py --sdaa-log sdaa.log --cuda-log cuda.log