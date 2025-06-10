# SENet154

## 1. 模型概述
CenterNet使用关键点检测的方法去预测目标边框的中心点，然后回归出目标的其他属性，例如大小、3D位置、方向甚至是其姿态。而且这个方向相比之前的目标检测器，实现起来更加简单，推理速度更快，精度更高。
- 仓库链接：[CenterNet](https://github.com/xingyizhou/CenterNet)

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 获取数据集
- 获取数据集。
用户可自行获取Pscal VOC数据集，将数据集上传到服务器任意路径下并解压；也可以通过下述脚本进行数据集的获取。

     - 运行脚本：

       ~~~
       cd ./src/tools/
       bash get_pascal_voc.sh
       ~~~

     - 上述脚本内容包含：

       - 从VOC网站下载、解压缩和移动Pascal VOC图像。
       - 下载COCO格式的Pascal VOC注释（从Detectron下载）。
       - 将train/val 2007/2012注释文件合并到单个json中。

    数据集目录结构参考如下所示。

       ```
       |-- data
       |-- |-- voc
           |-- |-- annotations
               |   |-- pascal_trainval0712.json
               |   |-- pascal_test2017.json
               |-- images
               |   |-- 000001.jpg
               |   ......
               |-- VOCdevkit        
       ```

    > **说明：**
    > 该数据集的训练过程脚本只作为一种参考示例。

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/build-in/Detection/CenterNet
    ```
2. 由于TecoPyTorch api `convolution_backward`默认使用硬件的ACE单元，会转成half来计算，精度有可能不够，导致模型无法收敛。
需要使用SIMD单元来开启纯fp32计算，训练前添加环境变量：
    ```
    export TORCH_SDAA_CONV2D_BACKWARD_USE_FP32=1
    ```
3. 运行训练脚本。

   该模型支持单机单卡训练。

    - 启动单卡训练。
      ```
      bash ./train_full_4p.sh --data_path=数据集路径  # 单卡4进程
      ```

   --data_path参数填写数据集路径，需写到数据集的一级目录。

   训练完成后，权重文件保存在`exp/`路径下，并输出模型训练精度和性能信息。

   更多训练参数参考 main.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./loss.py)，日志在`exp/`目录下）:<br>
MeanRelativeError: 0.011047458106808175
MeanAbsoluteError: 0.03403569999999996
Rule,mean_relative_error 0.011047458106808175
pass mean_relative_error=np.float64(0.011047458106808175) <= 0.05 or mean_absolute_error=np.float64(0.03403569999999996) <= 0.0002
