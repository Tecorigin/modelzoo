# FasterNet

## 1. 模型概述

FasterNet是一种高效的深度卷积神经网络模型，以其快速处理图像数据的能力而命名，该模型于 CVPR 2023上提出。模型通过采用混合尺寸卷积核和深度可分离卷积技术，减少了参数量和计算复杂度，同时保持了高识别精度。FasterNet利用多尺度特征融合和快速下采样，优化了信息流动和计算效率，适用于不同场景的模型变体包括FasterNet-T、FasterNet-S、FasterNet-M和FasterNet-L。

## 2. 快速开始

使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

FasterNet运行在ImageNet数据集上，数据集配置可以参考 https://blog.csdn.net/weixin_45783996/article/details/123738295

ImageNet正确解压后文件结构如下

```bash
ImageNet/
├── ILSVRC2012_devkit_t12
│   ├── data
│   │   ├── annotations
│   │   │   ├── val.txt            # 验证集图像的标注文件
│   │   │   └── ...               # 其他标注文件
│   │   └── ...
│   └── ...
├── train
│   ├── n01440764                 # 类别文件夹，通常以编号命名
│   │   ├── n01440764_10026.jpg   # 类别中的图像文件
│   │   ├── n01440764_10027.jpg
│   │   └── ...
│   ├── n01443537
│   │   ├── n01443537_10000.jpg
│   │   └── ...
│   └── ...                        # 其他类别文件夹
├── val
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   └── ...                        # 验证集图像文件
└── meta.mat
```

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**配置Python环境**

```bash
pip install -r requirements.txt

cd <modelzoo-dir>/Pytorch/contrib/Classification/FasterNet/scripts
```

**单机单核组**

```Python
torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29502 train_on_imagenet.py /data/datasets/imagenet/ -a fasternet -b 64 > ./scripts/output.log 2>&1 &
```

**单机单卡**

```Python
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29502 train_on_imagenet.py /data/datasets/imagenet/ -a fasternet -b 64 > ./scripts/output.log 2>&1 &
```

### 2.4 训练结果
模型训练2h共9个epoch，训练结果如下  

| 加速卡数量 |     模型      | 混合精度 | Batch Size | Epoch | train_loss | AccTop1 |
| :--------: | :-----------: | :------: | :--------: | :---: | :--------: | :-----: |
|     4      | FasterNet |   是    |     64     |   9   |   2.82    |  38.8%  |