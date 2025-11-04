# ViT-SAM
## 1. 模型概述
ViT-SAM 是基于 Meta（Facebook）的SAM（Segment Anything Model） 和 Vision Transformer（ViT） 结合的通用图像分割模型，旨在通过ViT强大的全局建模能力增强SAM的零样本（zero-shot）分割性能。其核心思想是 利用纯Transformer架构替代SAM中的CNN+Transformer混合设计，实现更高效的长距离上下文建模和更灵活的多尺度分割。
- 论文链接：[[2206.00272\]]An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale(https://arxiv.org/abs/2010.11929)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/vision_transformer
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
ViT-SAM使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/vit_sam/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_vit.py --config ../configs/vision_transformer/vit-base-p16_32xb128-mae_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: 4.018059671498722e-05
MeanAbsoluteError: 0.0002775050625942721
Rule,mean_relative_error 4.018059671498722e-05
pass mean_relative_error=4.018059671498722e-05 <= 0.05 or mean_absolute_error=0.0002775050625942721 <= 0.0002