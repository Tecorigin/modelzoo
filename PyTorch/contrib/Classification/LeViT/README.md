# LeViT

## 1.模型概述
**LeViT（Leaked Vision Transformer）** 是一种专为高效推理设计的轻量级视觉Transformer模型，由Facebook AI提出，旨在结合CNN的速度优势和Transformer的全局建模能力。  

LeViT的核心创新包括：  
1. **混合架构**：在Transformer基础上引入CNN风格的层次化结构，通过逐步降低分辨率并增加通道数，提升计算效率。  
2. **注意力改进**：采用**注意力偏置（Attention Bias）**替代传统的位置编码，简化计算并增强局部特征提取能力。  
3. **轻量化设计**：通过蒸馏训练（使用RegNet作为教师模型）和结构优化（如减少层数），显著降低参数量和计算成本。  

LeViT在ImageNet分类任务中表现优异，推理速度可比肩MobileNet等轻量CNN，同时保留Transformer的全局建模优势。其适用于移动端和边缘设备，为实时视觉任务（如分类、检测）提供了高精度、低延迟的解决方案。

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

LeViT运行在ImageNet数据集上，数据集配置可以参考
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
    --use_env main.py \
    --model LeViT_128 \
    --data-path /data/datasets/imagenet \
    --output_dir /data/ckpt \
    2>&1 | tee scripts/train_sdaa_3rd.log
```

### 2.4 训练结果

模型训练2小时左右后，得到结果如下：

| 加速卡数量 | 模型 | 混合精度 | Batch Size | Epoch/Iterations | Train Loss | AccTop1 |
|---|---|---|---|---|---|---|
| 1 | LeViT | Amp | 64 | Iterations：2800/5004 | 6.9557 | / |