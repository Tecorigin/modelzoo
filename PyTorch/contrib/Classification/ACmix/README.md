# ACmix

## 1. 模型概述
ACmix 是一种结合了卷积神经网络（CNN）和自注意力机制的深度学习模型，特别优化了计算效率与精度，广泛应用于图像处理与分类任务。通过融合先进的模型设计与分布式训练技术，ACmix 提供了强大的性能，在处理大规模数据集时具有较好的表现。

源码链接: https://github.com/LeapLabTHU/ACmix

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

ACmix运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/ACmix

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ACmix
    ```


- 单机单核组
    ```
  SDAA_LAUNCH_BLOCKING=1 python train.py --batch_size 64 --epochs 1 --distributed False --dataset_path /data/datasets/imagenet
  --lr 0.0005 --autocast True
    ```
- 单机单卡
    ```
   python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/bupt/datasets/imagenet \
   --batch_size 64 --epochs 1 --distributed True --lr 0.0005 --autocast True
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| ACmix |是|64|224*224|

