# RNN-Time-series-Anomaly-Detection

## 1. 模型概述
RNN-Time-series-Anomaly-Detection是一个基于RNN（循环神经网络）的时间序列异常检测模型，采用两阶段策略：首先在无异常的训练数据上训练RNN进行多步时间序列预测，然后在测试阶段通过比较实际值与预测值的残差，并结合多元高斯分布计算异常分数，从而识别出异常点。该实现使用PyTorch，适用于多种单变量或多元时间序列数据集。


- 参考实现：
    ```
    url=https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection
    commit_id=21ddb78f9128cc5a96dfaf4b27ee4db76a1a68ee
    ```


## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 准备数据集

- 本实验采用nyc_taxi数据集进行训练。

- 请你执行以下命令下载数据集：

   ```
   python 0_download_dataset.py
   ```

- 下载后将数据集置于仓库的任意位置。

### 2.3 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.3.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/other/RNN-Time-series-Anomaly-Detection
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.3.2 执行以下命令，构建名为`sdaa_RNN-Time-series-Anomaly-Detection`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_RNN-Time-series-Anomaly-Detection
   ```

#### 2.3.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_RNN-Time-series-Anomaly-Detection -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.3.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_RNN-Time-series-Anomaly-Detection /bin/bash
    ```

#### 2.3.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.3.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```


### 2.4 启动训练

#### 2.4.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/other/RNN-Time-series-Anomaly-Detection
    ```

#### 2.4.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2下载数据集。
  - 启动训练：
    ```
    python 1_train_predictor.py --data ecg --filename xxx.pkl
    ```
    其中：filename参数可以指定保存的模型文件名。

### 2.5 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: -0.0022419884
MeanAbsoluteError: -0.004122137
Rule,mean_absolute_error -0.004122137
pass mean_relative_error=-0.0022419884 <= 0.05 or mean_absolute_error=-0.004122137 <= 0.0002


