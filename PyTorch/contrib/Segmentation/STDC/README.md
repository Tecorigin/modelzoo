# STDC

## 1. 模型概述

Short-Term Dense Concatenate network（STDC）是一种在计算机视觉领域颇具特色的神经网络架构，常用于语义分割等任务。它通过独特的短期密集连接机制，将网络不同层之间进行密集连接，使得信息能够在层与层之间高效传递和复用，充分利用了各层提取的特征。这种结构能够有效避免梯度消失问题，同时促进了特征的融合，增强了模型对图像特征的学习能力。在网络设计上，它合理地平衡了计算量和特征表达能力，以较小的参数量和计算成本实现了较好的性能表现。在语义分割任务中，STDC 能够准确地对图像中的不同物体和区域进行分割和分类，为诸如自动驾驶场景下的道路元素识别、医学图像中组织器官的分割等应用提供了有力支持。凭借其在特征复用和高效计算方面的优势，STDC 已成为计算机视觉领域中推动实时语义分割等任务发展的重要模型之一。

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
    cd PyTorch/contrib/Segmentation/STDC
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    sh tools/dist_train.sh ./configs/stdc/stdc2_4xb12-80k_cityscapes-512x1024.py 4 | tee ./scripts/train_sdaa_3rd.log
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
 2h训练了565个iter
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss |
| --- | --- | --- | --- | --- | --- |
| 4 | STDC | 是 | 8 | 565 | 9.2842 |