# SqueezeNet 

## 1. 模型概述
SqueezeNet 是一种轻量化的深度卷积神经网络模型，通过引入 Fire 模块显著减少了模型参数量和存储需求，同时保持了良好的分类性能。SqueezeNet 已被广泛应用于图像分类、目标检测等计算机视觉任务。
源码链接:https://github.com/forresti/SqueezeNet 

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

SqueezeNet 运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/SqueezeNet 

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/SqueezeNet 
    ```


- 单机单核组
    ```
   python train.py --batch_size 128 --epochs 11 --distributed False --dataset_path /data/datasets/imagenet
  --lr 0.04 --autocast True 
    ```
- 单机单卡
    ```
  python -m torch.distributed.launch --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet \
  --batch_size 128 --epochs 11 --distributed True --lr 0.04 --autocast True
   ```

### 2.5 训练结果



| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| SqueezeNet  |是|128|224*224|
