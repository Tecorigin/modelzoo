# EfficientNet

## 1. 模型概述
EfficientNet 是由谷歌的研究团队在 2019 年提出的一种新型卷积神经网络（CNN）架构，旨在优化模型的性能和效率。其主要目标是通过减少计算成本和参数数量，同时提高模型的准确性和可扩展性。

源码链接: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test9_efficientNet

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

EfficientNetB3运行在ImageNet数据集上，数据集配置可以参考https://blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/EfficientNetB3

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```

2. 安装python依赖
``` 
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/EfficientNetB3
    ```
训练过程保存的best权重会保存在"examples/imagenet/model_best.pth.tar"中。

- 单机单核组
    ```
    !cd ../examples/imagenet
    torchrun --nproc_per_node=1 mmain.py --arch efficientnet-b3 -b 32
    ```
- 单机单卡
    ```
    !cd ../examples/imagenet
    torchrun --nproc_per_node=4 mmain.py --arch efficientnet-b3 -b 32
    ```


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| EfficientNet-B3 |是|32|300*300|




