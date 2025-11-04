###  Deep Sort

**1.模型概述** 

DeepSort 是一种基于深度学习的多目标跟踪算法，它在经典的 SORT（Simple Online and Realtime Tracking）算法基础上引入了深度学习模型来提取目标的外观特征，从而提高了跟踪的准确性和鲁棒性。其核心流程包括目标检测、特征提取、轨迹预测与匹配等步骤。

**2.快速开始**

使用本模型执行训练的主要流程如下：

基础环境安装：介绍训练前需要完成的基础环境检查和安装。

获取数据集：介绍如何获取训练所需的数据集。

启动训练：介绍如何运行训练。

**2.1 基础环境安装**

注意激活自身环境
（注意克隆torch.sdaa库）

**2.2 获取数据集**


Market-1501-v15.09.15数据集可以在官网进行下载；


**2.3 启动训练**

运行脚本在当前文件下，该模型在可以支持4卡分布式训练

1.cd到指定路径下，源码是一个比较大的框架（根据实际情况）

    cd ..

    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

2.运行指令
加载预训练模型ckpt.t7进行训练

**单机单核组**

    python ./deep_sort/deep/train.py --data-dir /data/datasets/Market-1501-v15.09.15 --weights ./deep_sort/deep/checkpoint/ckpt.t7

**单机单卡**

    SDAA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 ./deep_sort/deep/train_multiGPU.py --data-dir /data/datasets/Market-1501-v15.09.15 --weights  ./deep_sort/deep/checkpoint/ckpt.t7 

**2.4 训练结果**

| 加速卡数量 | 模型      | 混合精度   | Batch_Size | Accuracy |
|---|---|---|---|---|
|1 | DeepSort  | 是  | 32 | 0.74 |