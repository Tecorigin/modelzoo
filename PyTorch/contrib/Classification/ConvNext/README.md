# ConvNeXt

## 1. 模型概述
ConvNeXt 是一种基于卷积的现代化神经网络架构，结合了卷积神经网络（CNN）的高效性和 Transformer 的自注意力机制。ConvNeXt 通过改进卷积操作，优化网络架构设计，提高了计算效率和精度，广泛应用于图像分类任务。

源码链接: https://github.com/facebookresearch/ConvNeXt-V2

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

ConvNeXt运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/ConvNeXt

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```

2. 安装python依赖
``` 
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ConvNeXt
    ```


- 单机单核组
    ```
    SDAA_LAUNCH_BLOCKING=1 python train.py --batch_size 64 --epochs 1 --distributed False --dataset_path /data/datasets/imagenet
     --lr 0.00015 --autocast True
    ```
- 单机单卡
    ```
   python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
    --batch_size 64 --epochs 1 --distributed True --lr 0.00015 --autocast True
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| ConvNeXt |是|64|224*224|


