# Milan
## 1. 模型概述
Milan是一个由谷歌开发的多模态大模型，其核心特点是能够对图像进行极其精细和全面的描述（Rich Image Captioning）。

- 论文链接：[[2403.15360]]Milan: Masked Image Pretraining for Language Image Representation(https://arxiv.org/abs/2403.15360)
- 仓库链接：https://github.com/open-mmlab/mmpretrain/tree/main/configs/milan


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
Milan使用 ImageNet 数据集，该数据集为开源数据集，可从 [ImageNet](https://image-net.org/) 下载。

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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/MILAN/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
python run_milan.py --config ../configs/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k.py --launcher pytorch --nproc-per-node 1 --amp | tee sdaa.log

   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

MeanRelativeError: -0.005533788008300723
MeanAbsoluteError: -0.00786734689580332
Rule,mean_absolute_error -0.00786734689580332
pass mean_relative_error=np.float64(-0.005533788008300723) <= 0.05 or mean_absolute_error=np.float64(-0.00786734689580332) <= 0.0002