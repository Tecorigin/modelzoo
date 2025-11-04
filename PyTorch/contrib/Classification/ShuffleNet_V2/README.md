# ShuffleNet_v2
## 1. 模型概述
ShuffleNetV2 是由旷视科技（Megvii） 在 2018年提出的高效轻量级卷积神经网络，是 ShuffleNetV1 的改进版本。该模型针对 移动端和嵌入式设备 优化，通过重新设计网络结构和优化计算方式，在 速度-精度权衡 上表现优异。

- 论文链接：[title]ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design(https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/shufflenet_v2
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
ShuffleNetV2使用 ImageNet数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ShuffleNet_V2/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
   python run_shufflenet_v2.py --config ../configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py \
    --launcher pytorch --nproc-per-node 4 --amp \
    --cfg-options "train_dataloader.dataset.data_root=$data_path" "val_dataloader.dataset.data_root=$data_path" 2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.0020213101568084544
MeanAbsoluteError: -0.014560269837332244
Rule,mean_absolute_error -0.014560269837332244
pass mean_relative_error=-0.0020213101568084544 <= 0.05 or mean_absolute_error=-0.014560269837332244 <= 0.0002