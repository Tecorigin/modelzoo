# AlexNet
## 1. 模型概述
AlexNet是一种深度卷积神经网络模型，由Alex Krizhevsky、Ilya Sutskever和 Geoffrey E. Hinton在2012年提出，并在当年的ImageNet大规模视觉识别挑战赛（ILSVRC）中取得了第一名的成绩。AlexNet的成功标志着深度学习在计算机视觉领域的崛起，并推动了该领域后续的发展。
## 2. 快速开始
使用本模型执行训练的主要流程如下：  
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。  
2. 获取数据集：介绍如何获取训练所需的数据集。  
3. 启动训练：介绍如何运行训练。  
### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。
### 2.2 准备数据集
AlexNet运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729
### 2.3 启动训练
该模型支持单机单核组、单机单卡  
**单机单核组**
```Python
torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a alexnet -b 256
```
**单机单卡**
```Python
torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=29500 main.py /data/datasets/imagenet -a alexnet -b 256
```
### 2.4 训练结果
模型训练2h，得到结果如下  
|加速卡数量|模型|混合精度|Epoch|Batch size|AccTop1|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |   
|1|AlexNet|是|8|256|18.956%|4.183|
