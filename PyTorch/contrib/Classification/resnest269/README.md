###  Resnest269 


**1.模型概述** 

ResNeSt是一种改进版的ResNet模型，全称为Split-Attention Networks，即分注意力网络。它通过引入Split-Attention模块，将特征图沿通道维度划分为多个组和细粒度的子组，每个组的特征表示通过其子组表示的加权组合来确定，权重基于全局上下文信息选择。这种设计在不增加额外计算成本的情况下，显著提升了模型对局部和全局特征的捕捉能力。ResNeSt在多个任务上表现出色，如在ImageNet上，ResNeSt-50的top-1准确率达到了81.13%。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境

**2.2 获取数据集**

Imagenet数据集可以在官网进行下载；


**2.3 启动训练**

运行脚本在scripts文件下，该模型在可以支持4卡分布式训练

（1） cd 到指定的目录下，注意进入到当前模型所在位置

    cd ..
    pip install resnest --pre
    pip install -r requirements.txt

（2） 开始训练，每个模型均有一个对应的config文件对应；

**单机单核组**

由于使用了 torch.sdaa.device_count()函数
如果需要跑单卡，可以在train.py第80行 ngpus_per_node = torch.sdaa.device_count()进行修改，修改为1
    

    nohup python -u ./scripts/torch/train.py --config-file ./configs/config269.yaml >>resnest.txt 2>&1 &

**单机单卡组**

    nohup python -u ./scripts/torch/train.py --config-file ./configs/config269.yaml >>resnest.txt 2>&1 &

**2.4 训练结果**
|加速卡数量| 模型  |  混合精度 | Batch_Size  |  Shape |  AccTop1 |
|---|---|---|---|---|---|
|  1 | resnest269 |  是 |  8  |  416*416 |  0.06 |


