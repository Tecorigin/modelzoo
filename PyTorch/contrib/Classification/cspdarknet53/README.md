# cspdarknet53

## 1. 模型概述
CSPDarknet53 是一种深度卷积神经网络模型，主要用于图像分类、目标检测和其他计算机视觉任务。它是在 Darknet53 的基础上进行改进的，采用了 CSPNet（Cross-Stage Partial Networks）方法来提高模型的效率和性能。

源码链接: https://github.com/njustczr/cspdarknet53

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备

#### 2.2.1 数据集位置
/data/datasets/imagenet


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
``` bash
cd <ModelZoo_path>/PyTorch/contrib/Classification/cspdarknet53

conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```
<p align="center">
    <img src="image/env.png" alt="Source Image" width="80%">
</p>

2. 安装python依赖
``` 
bash
# install requirements
pip install -r requirements.txt
```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/cspdarknet53
    ```
2.每个iteration的loss图会保存在/output路径中

- 单机单核组
    ```
    python main.py /data/datasets/imagenet
    ```
- 单机单卡
    ```
    python main.py /data/datasets/imagenet --gpu 0 --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:65501' --world-size 1 --rank 0 --dist-backend 'tccl' --workers 40
    ```

更多训练参数参考[README](run_scripts/README.md)


### 2.5 训练结果

| 芯片 |卡  | 模型 |  混合精度 |Batch size|Shape| 
|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1| cspdarknet53 |是|32|300*300|

**训练结果量化指标如下表所示**

| 训练数据集 | 输入图片大小 |nproc_per_node|accuracy(DDP)|
| :-----: | :-----: |:------: |:------: |
| mini_imagenet | 224x224 |4|84.0% |

**训练过程loss曲线如下图所示**
<p align="center">
    <img src="image/loss_curve.png" alt="Source Image" width="55%">
</p>

