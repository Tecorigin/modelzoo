# SqueezeNext

## 1.模型概述

SqueezeNext 是一种轻量级卷积神经网络架构，专为移动和嵌入式设备设计，注重高效计算与低功耗。其核心创新在于采用分离式Fire模块（Split-Fire Module），将传统卷积分解为1x1和可分离卷积，大幅减少参数量。结合瓶颈结构与跳跃连接优化信息流动，在保持精度的同时降低计算复杂度。  

SqueezeNext 在 ImageNet 分类任务中表现接近 MobileNet，但计算量更少（如 SqueezeNext-23 仅需 0.72 亿次乘加操作/图像）。支持模型压缩技术（如量化、剪枝），适合部署在资源受限的边缘设备，平衡了性能与能效。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/SqueezeNext
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

SqueezeNext运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**单机单核组**

```
python train.py --dataset_path /data/datasets/imagenet --batch_size 16 --epochs 20 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet --batch_size 16 --epochs 3 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

模型训练2h，得到结果如下

| 加速卡数量 |    模型     | 混合精度 | Batch Size | Epoch | train_loss | AccTop1 |
| :--------: | :---------: | :------: | :--------: | :---: | :--------: | :-----: |
|     1      | SqueezeNext |   amp    |     16     |   3   |   5.394    |  0.099  |

