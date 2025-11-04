# APCNet

## 1. 模型概述

APCNet 即 Adaptive Pyramid Context Network，是一种应用于计算机视觉领域的神经网络模型，采用了自适应金字塔池化模块和上下文信息融合等结构。它通过自适应地对输入特征图进行多尺度池化操作，有效捕捉不同大小物体和场景的上下文信息，并利用特征增强机制提升特征表现力，以解决语义分割等任务中的多尺度目标适应和上下文理解问题。APCNet 在语义分割、目标检测和图像理解与分析等计算机视觉任务中表现出色，凭借多尺度特征利用、强大的上下文信息捕捉能力以及良好的模型灵活性和可扩展性等优势，已成为该领域的重要模型之一。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 Cityscapes 数据集。链接：https://www.cityscapes-dataset.com/

#### 2.2.2 处理数据集

- 解压训练数据集：
- 现在/data/datasets/cityscapes/中包含已经转换好的数据集，因此可以直接使用，无需运行下述命令。
``` bash
cd /data/datasets/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
python -m pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py /data/datasets/cityscapes/  
```

#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
Cityscapes/
|-- gtFine
|   |-- test
|   |-- train
|   `-- val
`-- leftImg8bit
    |-- test
    |-- train
    `-- val
```


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动环境。
参考[官方教程](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)进行安装
```sh
conda create -n mmseg--clone torch_env
conda activate mmseg
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
- 若在安装mmengine时报错："GLIBCXX not found"
参考以下[博客](https://zhuanlan.zhihu.com/p/685165815)和代码解决：
```sh
cd /root/anaconda3/envs/mmseg/lib/ 
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  .
ln -s -f libstdc++.so.6.0.30 libstdc++.so.6
```
- 注意： mmseg库需要修改site-package中的mmengine环境依赖。建议直接clone迁移好的mmseg环境。

2. 下载代码
``` 
git clone https://github.geekery.cn/https://github.com/open-mmlab/mmsegmentation.git
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
    cd PyTorch/contrib/Segmentation/APCNet
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=2,3
    sh tools/dist_train.sh configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py 2 
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
 2h训练了3700个iter
 | 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss | AccTop1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | APCNet | 否 | 2 | 3800 | 0.7023 | - | 