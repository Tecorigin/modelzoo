# RepViT

## 1.模型概述
RepViT 是一种新颖的混合模型，它结合了卷积神经网络（CNNs）和Vision Transformer的优势，旨在实现高性能和高效率。RepViT 的核心思想是在 Vision Transformer 的基础上，利用 CNNs 的局部感受野和 Transformer 的全局感受野，实现特征的有效提取和融合。RepViT模型通过精心设计的混合架构，在图像分类、目标检测和语义分割等任务上取得了出色的表现。 RepViT 是一种兼顾性能和效率的卷积神经网络模型，这使得它在资源受限的应用场景中具有很大的优势。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

RepViT运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

模型单机单卡训练：
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 65501 --use_env main.py --model repvit_m0_9 --dist-eval --batch-size 16 --epochs 1 2>&1|tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练总耗时为2小时，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | RepViT | Amp | 16 | Iterations：9100/40032 | 6.9755 | 0.142 |