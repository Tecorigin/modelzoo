# PVT_V2

## 1. 模型概述
PVTv2（Pyramid Vision Transformer v2）是一种改进型的视觉Transformer模型，结合金字塔结构与卷积增强机制，在提升建模能力的同时显著优化了推理效率。PVT v2 通过引入更高效的注意力机制和结构设计，增强了多尺度特征表示能力，适用于图像分类、目标检测、语义分割等多种计算机视觉任务，在保持较低计算复杂度的同时取得了更优的性能表现。
源码链接:https://github.com/whai362/PVT/blob/v2/detection

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

PVTv2运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/PVT_V2 

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/PVT_V2 
    ```


- 单机单核组
    ```
   python train.py --batch_size 128 --epochs 10 --distributed False --dataset_path /data/datasets/imagenet
  --lr 0.0001 --autocast True 
    ```
- 单机单卡
    ```
  python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
  --batch_size 128 --epochs 10 --distributed True --lr 0.0001 --autocast True
   ```

### 2.5 训练结果



| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| PVTv2 |是|128|224*224|
