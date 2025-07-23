# DeiT

## 1.模型概述
**DeiT（Data-efficient Image Transformer）模型**是一种基于Transformer架构的高效图像分类模型，由Facebook AI团队提出，旨在解决Vision Transformer（ViT）对大规模数据依赖的问题。DeiT通过引入**知识蒸馏**和**数据增强**策略，仅使用ImageNet数据集（无需额外的预训练数据）即可达到与ViT相当甚至更好的性能。  

DeiT的核心创新包括：  
1. **师生蒸馏**：利用预训练的CNN（如RegNet）作为教师模型，指导DeiT训练，显著提升小模型的表现。  
2. **高效训练策略**：通过优化超参数和增强数据（如RandAugment、MixUp），减少对海量数据的依赖。  
3. **轻量化设计**：提供多种尺寸的模型（如DeiT-Tiny、DeiT-Small），适合不同计算资源场景。  

DeiT在ImageNet分类任务中表现优异，尤其适合资源受限的场景。其平衡了Transformer的全局建模能力和训练效率，为视觉任务的轻量化部署提供了新思路。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

DeiT运行在ImageNet数据集上，数据集配置可以参考
https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 启动训练

准备环境：
```
pip install -r requirements.txt
```

模型单机单卡训练：
```shell
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 65501 \
    --use_env main.py \
    --model deit_tiny_patch16_224 \
    --batch-size 256 \
    --data-path /path/to/imagenet \
    --output_dir /path/to/ckpt \
    2>&1 | tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练2小时左右后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | DeiT | Amp | 256 | Epoch：3 | 6.6483 | 3.306 |