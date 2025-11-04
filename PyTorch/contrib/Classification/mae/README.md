
# MAE
## 1. 模型概述
掩模自编码器 (MAE) 是可扩展的计算机视觉自监督学习器。它基于两个核心设计：非对称编码器-解码器架构，其中编码器仅对可见的块子集（不含掩码标记）进行操作，以及一个轻量级解码器，该解码器根据潜在表示和掩码标记重建原始图像，另外掩码输入图像的很大一部分（例如 75%）可以产生一个非平凡且有意义的自监督任务。结合这两种，能够高效且有效地训练大型模型：将训练速度提高（3 倍或更多）并提高准确率。这个方法方法允许学习具有良好泛化能力的大容量模型：例如，在仅使用 ImageNet-1K 数据的方法中，原始 ViT-Huge 模型实现了最佳准确率（87.8%）。下游任务中的转移性能优于监督预训练，并表现出有希望的扩展行为。

- 论文链接：[2111.06377\]Masked Autoencoders Are Scalable Vision Learners(https://arxiv.org/abs/2111.06377)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/mae

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Deit 使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/xzxg001/article/details/142465729。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip3 install  -U openmim 
    pip3 install git+https://gitee.com/xiwei777/mmengine_sdaa.git 
    pip3 install opencv_python mmcv --no-deps
    mim install -e .
    pip install -r requirements.txt

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/mae/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_mae.py --config ../configs/mae/mae_hivit-base-p16_8xb512-amp-coslr-400e_in1k.py \
    --launcher pytorch --nproc-per-node 1 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path"  2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./image/loss.jpg)

MeanRelativeError:0.012161317861653646
MeanAbsoluteError:0.02052541770557366
Rule,mean_relative_error0.012161317861653646
pass mean_ralative_error=0.012161317861653646 <=0.05 or  mean_absolute_error=0.02052541770557366<=0.0002


