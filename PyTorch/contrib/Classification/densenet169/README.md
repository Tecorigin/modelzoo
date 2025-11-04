# DenseNet
## 1. 模型概述
DenseNet是一种深度卷积神经网络模型，它通过引入Dense Connectivity的结构，解决了传统卷积网络中信息和梯度在深层传播时容易丢失的问题。DenseNet的每一层与其之前的所有层直接相连，这种连接方式不仅加强了特征传播、增强了特征再利用，还减轻了过拟合现象，并且相对于其他同样深度的网络减少了参数数量。  
DenseNet模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其独特的架构设计，DenseNet能够在更少的参数下实现更高的准确率，这使得它在资源受限的应用环境中特别有价值。凭借其出色的性能和广泛的应用场景，DenseNet已成为深度学习和计算机视觉领域的重要基础模型之一。 当前支持的DenseNet模型包括：DenseNet121、DenseNet161、DenseNet169和DenseNet201。
## 2. 快速开始
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  
### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2 准备数据集
DenseNet运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729
### 2.3 启动训练
该模型支持单机单核组、单机单卡  
**单机单核组**
```Python
torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a densenet169 -b 64
```
**单机单卡**
```Python
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29501 main.py /data/datasets/imagenet -a densenet169 -b 64
```
### 2.4 训练结果
模型训练2h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|AccTop1|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|1|DenseNet169|是|2|64|17.270%|4.303|
