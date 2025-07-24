
# CSRA
## 1. 模型概述
CSRA（Class-Specific Residual Attention）是由南京大学（Nanjing University）视觉几何组提出的一种简洁而高效的多标签图像识别（Multi-Label Image Recognition）模型模块。其核心思想是通过引入一种类别特定的空间注意力机制，为每个类别生成具有区分性的特征表示，并将其与类别无关的全局平均池化特征相结合，从而更有效地捕捉图像中属于不同类别的空间区域。CSRA在保持模型结构极其简单的同时（仅需4行代码实现），在多个主流多标签数据集上取得了最先进的识别性能（state-of-the-art results），并且无需额外训练即可显著提升不同预训练模型的表现。作为一种轻量级、通用性强、计算代价低的注意力机制，CSRA为多标签识别任务提供了一个直观易实现的新思路，也成为该领域的代表性方法之一。


- 论文链接：[[2108.02456v2\]]Residual Attention: A Simple but Effective Method for Multi-Label Recognition(https://arxiv.org/pdf/2108.02456)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/csra

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
csra 使用 VOC2007 数据集，该数据集为开源数据集，

#### 2.2.2 处理数据集
具体配置方式可参考：https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/dataset_prepare.html


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
  cd <ModelZoo_path>/PyTorch/contrib/Classification/csra/run_scripts
  ```

2. 运行训练。该模型支持单机单卡。
  ```
  python run_csra.py --config ../configs/csra/resnet101-csra_1xb16_voc07-448px.py --launcher pytorch --nproc-per-node 4 --amp --cfg-options "train_dataloader.dataset.data_root=<imagenet_path>" "val_dataloader.dataset.data_root=<imagenet_path>"
  ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
<table>
  <tr>
    <th>模型</th>
    <th>数据集</th>
    <th colspan=3>sdaa结果</th>
    <th colspan=3>cuda结果</th>
    <th>sdaa耗时</th>
  </tr>
  <tr>
    <td rowspan=2>csra</td>
    <td rowspan=2>VOC2007</td>
    <td>CF1</td>
    <td>OF1</td>
    <td>mAP</td>
    <td>CF1</td>
    <td>OF1</td>
    <td>mAP</td>
    <td rowspan=2>1h</td>
  </tr>
  <tr>
    <td>89.1677</td>
    <td>91.2342</td>
    <td>94.1687</td>
    <td>89.16</td>
    <td>90.80</td>
    <td>94.98</td>
  </tr>
</table>
    

