
# Demo
## 1. 模型概述
基于 CNN 的 PyTorch 实现，用于面部表情识别（FER2013 和 CK+），在 FER2013 数据集上达到 72.112%（当前最佳水平），在 CK+ 数据集上达到 94.64%。
https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch?tab=readme-ov-file

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集
#### 2.2.1 获取数据集
数据集来自https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data 图像属性：48 x 48 像素（2304 字节） 标签：0=愤怒，1=厌恶，2=恐惧，3=高兴，4=悲伤，5=惊讶，6=中性 训练集包含 28,709 个样本。公开测试集包含 3,589 个样本。私有测试集包含另外 3,589 个样本。

#### 2.2.2 处理数据集
> 首先下载数据集（fer2013.csv）然后将其放入“data”文件夹，然后
python preprocess_fer2013.py


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```

>
2. 安装python依赖
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Demo/run_scripts
    ```

2. 运行训练。该模型支持单机单卡.

    -  单机单卡
    ```
   python run_demo.py 
    ```

    

### 2.5 训练结果
loss对比曲线: 
![loss对比曲线](./loss_comparison.png)

best_PublicTest_acc: 70.967
best_PublicTest_acc_epoch: 148
best_PrivateTest_acc: 72.471
best_PrivateTest_acc_epoch: 155

> 如果为完整的训练或微调任务，请提供最终的metric结果。
> 如果为短训，请提供loss曲线图和最终的loss结果。