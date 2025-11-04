# VGG19
## 1. 模型概述
VGG是一种深度卷积神经网络模型，它以牛津大学的视觉几何组（Visual Geometry Group）的名字命名，该团队在2014年提出了这种模型。VGG的主要特点是其简单且统一的架构，其中只使用了3x3的小型卷积核，并通过堆叠多个这样的卷积层来增加网络深度，同时在网络的不同阶段逐渐减少特征图的数量并增加卷积核的数量。VGG模型还包含多个最大池化层用于降采样。当前支持的VGG模型包括：VGG11、VGG13和VGG19。
## 2. 快速开始
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  
### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2 准备数据集
VGG运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729
### 2.3 启动训练
该模型支持单机单核组、单机单卡  
**单机单核组**
```Python
torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a vgg19 -b 256
```
**单机单卡**
```Python
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29501 main.py /data/datasets/imagenet -a vgg19 -b 256
```
### 2.4 训练结果
模型训练2h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|AccTop1|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|1|VGG19|是|4|256|5.717%|5.366|
