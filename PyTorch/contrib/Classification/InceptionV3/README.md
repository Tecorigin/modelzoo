# InceptionV3



## 介绍
InceptionV3模型是一种深度卷积神经网络模型，该模型通常用于各种图像分类任务、目标检测、图像分割等应用中，并在许多计算机视觉竞赛中取得了优异的成绩。

## 特征

-  **低侵入性**：易于集成和应用，无需对现有深度学习框架进行大幅修改。
-   **多尺度特征提取**：采用Inception模块，通过多种卷积核和池化操作组合，实现多尺度特征提取。
-   **计算效率高**：通过因式分解卷积和批量归一化等技术，显著降低计算复杂度，提高训练和推理效率。
-   **辅助分类器**：引入辅助分类器，增强梯度传播，提升模型性能并减少过拟合风险。
-   **全局平均池化**：使用全局平均池化层，减少参数数量，提高模型的泛化能力。
-   **预训练模型支持**：提供预训练模型，可用于快速进行微调，适用于各种图像分类任务。

## 快速指南
使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。

2. 获取数据集：介绍如何获取训练所需的数据集。

3. 启动训练：介绍如何运行训练。

## 运行环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。需要的依赖包在requirement.txt中。

1. 安装依赖

```jsx
cd <ModelZoo_path>/PyTorch/Classification/InceptionV4
```

## 准备数据集

###  拉取代码仓

```
git clone https://gitee.com/tecorigin/modelzoo.git
```

###  创建Teco虚拟环境

```
cd /modelzoo/PyTorch/Classification/InceptionV3
conda activate torch_env

# 执行以下命令验证环境是否正确，正确则会打印如下版本信息
python -c "import torch_sdaa"
```

```text
![环境信息](/env.jpg "env")
```

```
# install requirements
pip install -r requirements.txt

# install tcsp_dllogger
git clone https://gitee.com/xiwei777/tcap_dllogger.git
cd tcap_dllogger
python setup.py install
```

## 获取数据集

### 从百度网盘中下载数据集

InceptionV4运行在CIFAR-10上，这是一个广受欢迎的图像分类数据集。CIFAR-10数据集包含60000张32x32的彩色图像，分为10类，每类6000张。训练集包含50000张图像，测试集包含10000张图像。无需再次划分，下载自己需要的即可。

训练所用的示例CIFAR10数据集可以在百度云下载。

链接：https://pan.baidu.com/s/1wFpI4Kx2N_Os0QKNIO8FJw?pwd=kxa0 
提取码：kxa0

### 处理数据集

1. 下载好数据集，将训练用的CIFAR10分类数据集解压后放在根目录。

```
unzip cifar-10-batches-py.zip
```

文件格式如下：

```
|-- InceptionV3
    |-- checkpoint         # 权重放置地方
    |-- dataset               # 你需要的数据集，可选择下载
       |-- cifar-10-batches-py #数据集名称
	        |-- data_batch_1.bin        # 训练集1
	        |-- data_batch_2.bin        # 训练集2
	        |-- data_batch_3.bin        # 训练集3
	        |-- data_batch_4.bin        # 训练集4
	        |-- data_batch_5.bin        # 训练集5
	        |-- data_batch_6.bin        # 训练集6
	        |-- test_batch.bin          # 测试集
	        |-- batches.meta.txt
	        |-- batches.meta.txt
	        ├── readme.html
```

### 启动训练

训练命令：支持单机单SPA以及单机多卡（DDP）。训练过程保存的权重以及日志均会保存在"train"中。

- 单机单SPA训练

```jsx
python run_scripts/run_incv3.py --nproc_per_node 4  --model_name inceptionv3 --dataset_path "./dataset/cifar10"  --batch_size 128 --epoch 50 --device sdaa --autocast True --distributed False
```

- 单机多卡训练（DDP）

```jsx
python run_scripts/run_incv3.py --nproc_per_node 4  --model_name inceptionv3 --dataset_path "./dataset/cifar10"  --batch_size 128 --epoch 50 --device sdaa --autocast True --distributed True
```

训练命令参数说明参考[文档](https://gitee.com/i-cant-give-a-name/modelzoo/blob/VGG11/PyTorch/contrib/Classification/VGG11/run_scripts/README.md)。

### **2.4 训练结果**

#### 单卡

| 芯片 | 卡   | 模型        | 混合精度 | Batch size | Shape |
| ---- | ---- | ----------- | -------- | ---------- | ----- |
| SDAA | 1    | InceptionV3 | 是       | 128         | 32*32 |

**训练结果量化指标如下表所示**

| 训练数据集    | 基础模型    | 输入图片大小 | accuracy |
| ------------- | ----------- | ------------ | -------- |
| CIFAR10数据集 | InceptionV3 | 32x32        | 94.18%   |

#### 多卡

| 芯片 | 卡   | 模型        | 混合精度 | Batch size | Shape |
| ---- | ---- | ----------- | -------- | ---------- | ----- |
| SDAA | 4    | InceptionV3 | 是       | 512        | 32*32 |

**训练结果量化指标如下表所示**

| 训练数据集    | 基础模型    | 输入图片大小 | accuracy |
| ------------- | ----------- | ------------ | -------- |
| CIFAR10数据集 | InceptionV3 | 32x32        |16.2%   |

**单卡训练loss曲线:**

![训练loss曲线](/home/jiangnan/modelzoo/PyTorch/Classification/InceptionV3/img/Train.png)

**单卡测试准确率曲线：**

![训练准确率曲线](/home/jiangnan/modelzoo/PyTorch/Classification/InceptionV3/img/Test.png)

**多卡训练loss曲线:**

![训练loss曲线](/home/jiangnan/modelzoo/PyTorch/Classification/InceptionV3/img/train_multi.jpg)

**多卡测试准确率曲线：**

![训练准确率曲线](/home/jiangnan/modelzoo/PyTorch/Classification/InceptionV3/img/test_multi.jpg)

## **Reference**

https://github.com/weiaicunzai/pytorch-cifar100
