# ResNet34
## 1. 模型概述
ResNet是一种深度卷积神经网络模型，采用了残差网络（ResNet）的结构，通过引入残差块（Residual Block）以解决深度神经网络训练中的梯度消失和表示瓶颈问题。ResNet模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其良好的性能和广泛的应用，ResNet已成为深度学习和计算机视觉领域的重要基础模型之一。
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
torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a resnet34 -b 64
```
**单机单卡**
```Python
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a resnet34 -b 64
```
### 2.4 训练结果
模型训练2h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|AccTop1|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|1|ResNet34|是|7|64|44.231%|2.541|
