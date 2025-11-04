# SeResnet

## 1. 模型概述

SeResNet 即 Squeeze-and-Excitation ResNet，是在经典 ResNet 基础上融合了 Squeeze-and-Excitation（SE）模块的深度卷积神经网络模型。它通过 SE 模块对 ResNet 提取的特征进行通道维度上的重新校准，自动学习每个通道的重要性，对关键通道特征进行增强，抑制次要通道特征。这种结构使得 SeResNet 在图像分类、目标检测等计算机视觉任务中，能够更有效地捕捉特征间的依赖关系，提升模型的特征表示能力，进而提高任务的性能表现。凭借其在特征优化方面的优势，SeResNet 成为深度学习领域中优化网络性能的重要模型之一，广泛应用于各类需要高精度视觉处理的场景。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 ImageNet 数据集。按照如下链接准备：https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/dataset_prepare.html

#### 2.2.2 处理数据集
- 无需额外处理
- 数据集位置：`/data/datasets/20241122/imagenet1k/`


#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
imagenet1k/
|-- class_names
|-- meta
|-- train
`-- val

```


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动环境。

```sh
conda create -n mmpretrain --clone torch_env
conda activate mmpretrain
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
- 若在安装mmengine时报错："GLIBCXX not found"
参考以下[博客](https://zhuanlan.zhihu.com/p/685165815)和代码解决：
```sh
cd /root/anaconda3/envs/mmpretrain/lib/ 
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  .
ln -s -f libstdc++.so.6.0.30 libstdc++.so.6
```
- 注意： mmpretrain库需要修改site-package中的mmengine环境依赖。建议直接clone迁移好的mmpretrain环境。

2. 下载代码
``` 
git clone https://github.geekery.cn/https://github.com/open-mmlab/mmpretrain.git
```
3. 安装依赖
```
pip install -v -e .
pip install mmcv==2.1.0
pip install ftfy
pip install regex
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Classification/SeResnet101
    ```

2. 运行训练。
    ```
    SDAA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash ./tools/dist_train.sh configs/seresnet/seresnet101_8xb32_in1k.py 4
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./scripts
    python plot_curve.py
    ```
 2h训练了140个epoch
 | 加速卡数量 | 模型 | 混合精度 | Batch Size | epoch | train_loss |
| --- | --- | --- | --- | --- | --- |
| 4 | SE-Resnet101 | 是 | 32 | 100 | 3.8743 |