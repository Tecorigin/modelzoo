# MLP

## 1.模型介绍

多层感知器（MLP）是一种前馈神经网络，包含输入层、多个隐藏层和输出层。每个隐藏层中的神经元通过激活函数引入非线性，允许模型学习复杂的特征。

## 2. 快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装：

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 获取数据集：

MNIST数据集可以通过运行的代码自行下载无须提前准备

### 2.3 构建环境：

所使用环境包含PyTorch框架虚拟环境

1. 执行以下命令，启动虚拟环境。

   ```
   cd modelzoo/PyTorch/contrib/Classification/MLP/
   ```

   ```
   conda activate torch_env
   ```

   

2.安装python依赖

```
pip install -r requirements.txt
```



### 2.4 启动训练：

1.进入所需的环境目录

```
cd modelzoo/PyTorch/contrib/Classification/MLP/run_scirpts/
```

2.代码运行

单机单SPA训练：

```
python run_mlp.py  --device sdaa --epochs 10 --lr 0.001 --batch_size 64
```

单机多卡训练(DDP)：

```
python run_mlp.py --device sdaa --nproc_per_node 3 --epochs 10 --lr 0.001 --batch_size 64
```



### 2.5 训练结果

训练条件：

| 加速卡数量 | 模型 | 混合精度 | Batch size | Shape |
| ---------- | ---- | -------- | ---------- | ----- |
| 1          | MLP  | 是       | 64         | 28*28 |

结果展示：

| 训练数据集  | 基础模型 | 输入图片大小 | accuracy |
| ----------- | -------- | ------------ | -------- |
| MNIST数据集 | MLP      | 28*28        | 96.87%   |

