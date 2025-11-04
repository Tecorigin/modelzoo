
# segmenter
## 1. 模型概述
Segmenter是一种基于Vision Transformer（ViT）架构的语义分割模型，通过将图像分割为块序列并利用Transformer编码器-解码器结构实现像素级分类。

- 论文链接：[2105.05633  \]Segmenter: Transformer for Semantic Segmentation  (https://arxiv.org/abs/2105.05633  )
- 仓库链接：https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segmenter

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
 使用 ADE 数据集，该数据集为开源数据集，可从 (https://opendatalab.com/) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/advanced_guides/datasets.md。


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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/segmenter/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
    python run_segmenter.py --config ../configs/segmenter/segmenter_vit-b_mask_8xb1-160k_ade20k-512x512.py \
       --launcher pytorch --nproc-per-node 1 --amp 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 



MeanRelativeError: 0.22906344083738878
MeanAbsoluteError: -0.03253275480600867
Rule,mean_absolute_error -0.03253275480600867
pass mean_relative_error=0.22906344083738878 <= 0.05 or mean_absolute_error=-0.03253275480600867 <= 0.0002


