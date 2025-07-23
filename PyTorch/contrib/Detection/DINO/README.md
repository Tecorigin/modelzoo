# Dino

## 1. 模型概述
Dino 即 Distilling Representations with NO labels，是一种基于自监督学习的视觉模型。它开创性地采用基于注意力机制的 Transformer 架构，在没有标注数据的情况下，通过自监督的方式让模型学习图像的特征表示。Dino 利用多尺度特征聚合和动态掩码等技术，能够捕捉图像中丰富的语义信息和结构信息，从而生成高质量的特征表示。在图像分类、目标检测、语义分割等众多计算机视觉任务中，Dino 展现出了强大的性能，即使在没有大规模标注数据的情况下，也能学习到具有良好泛化能力的特征。凭借其在自监督学习领域的创新性和高效性，Dino 成为了深度学习和计算机视觉领域中推动自监督学习发展的重要模型之一，为解决数据标注难题和提升模型性能提供了新的思路和方法。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 ImageNet1K 数据集。链接：https://www.kaggle.com/c/imagenet-object-localization-challenge/data

#### 2.2.2 数据集目录结构

数据集目录结构参考如下所示:

```
imagenet1k/
|-- class_names
|-- meta
|-- train
`-- val
```


### 2.3 构建环境

1. 执行以下命令，启动虚拟环境。
``` bash
cd PyTorch/contrib/Detection/dino
conda activate dino
```
2. 安装python依赖
``` bash
pip install -r requirements.txt
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Detection/dino
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    cfg_file=configs/bisenetv2_city.py
    python -m torch.distributed.launch --nproc_per_node=4 --master_port 9292 --use_env main_dino.py
    ```


### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
| 加速卡数量 | 模型 | 混合精度 | Batch Size | epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 2 | Dino | 是 | 16 | 100 | 10.203 |