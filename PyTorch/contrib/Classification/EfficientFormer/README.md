# EfficientFormer

## 1. 模型概述
EfficientFormer 是一种高效的视觉变换器（Vision Transformer）架构，来自论文 EfficientFormer: Vision Transformers at MobileNet Speed，由 Yanyu Li 等人在 2022 年发表。EfficientFormer 通过优化 Transformer 结构，结合了池化操作（Pooling）和轻量化注意力机制（MetaFormer），显著降低了计算复杂度和参数量，同时保持了图像分类、目标检测和语义分割等任务的高性能。设计目标是实现 MobileNet 级别的推理速度，适合资源受限的设备（如移动设备和边缘设备）。原始代码仓库位于 https://github.com/snap-research/EfficientFormer。

本项目适配了 EfficientFormer 模型，提供在 PyTorch 框架下的训练和微调支持，适用于imagenet数据集下的分类任务等场景。

## 2. 快速开始
使用 EfficientFormer 模型执行训练的主要流程如下：
1. 基础环境安装：完成训练前的环境检查和安装。
2. 获取数据集：获取训练所需的数据集。
3. 构建环境：配置模型运行环境。
4. 启动训练：运行训练脚本。

### 2.1 基础环境安装
请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
EfficientFormer 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https：//image-net.org/) 下载。


#### 2.2.2 处理数据集
具体配置方式可参考：https：//blog.csdn.net/xzxg001/article/details/142465729

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录. 
```
cd <ModelZoo_path>/PyTorch/contrib/Classification/EfficientFormer/run_scripts
```
2. 运行训练. 该模型支持单机单卡。
```shell
python run_eformer.py --batch_size 128 --num_workers 4 --lr 0.05 --epoch 1 --num_steps 100 --device_num 1
```
更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 
![训练loss曲线](./run_scripts/loss.jpg)

MeanRelativeError: 0.0014996521963208693
MeanAbsoluteError: 0.009509143262806505
pass mean_relative_error=np.float64(0.0014996521963208693) <= 0.05 or mean_absolute_error=np.float64(0.009509143262806505) <= 0.0002