# MoCoV2

## 1. 模型概述
MoCoV2（Momentum Contrast V2）是一种基于对比学习的无监督视觉表示学习方法。MoCoV2 通过引入动量对比学习和增强的数据表示，显著提高了对比学习的效果，并且在多个图像分类任务中展示了较好的性能。相比于传统的对比学习方法，MoCoV2 采用了更高效的负样本存储和更新机制，并且优化了训练的稳定性和收敛速度。它在不使用标签的情况下，通过构建有效的正负样本对，学习到具有良好泛化能力的视觉特征。

源码链接: https://github.com/AidenDurrant/MoCo-Pytorch/tree/master

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

MoCoV2运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/MoCoV2

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/MoCoV2
    ```


- 单机单核组
    ```
  python main.py --no_distributed  --dataset imagenet --dataset_path /data/datasets/imagenet
    ```
- 单机单卡
    ```
   python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env main.py  --dataset imagenet 
  --dataset_path /data/datasets/imagenet
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|MoCoV2 |是|128|224*224|

