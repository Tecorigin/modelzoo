# deep-person-reid

## 1. 模型概述
deep-person-reid是一个基于 PyTorch 的开源深度学习库，专注于行人重识别（Person Re-Identification）任务。它由 Kaiyang Zhou 等人开发，集成了多种先进的 ReID 模型（如 OSNet、PCB、MGN 等），提供了完整的训练、评估和部署工具链，支持多数据集加载、多 GPU 训练、跨域识别和可视化分析等功能，旨在为学术研究和实际应用提供一个高效、模块化且易于扩展的开发平台。


- 参考实现：
    ```
    url=https://github.com/KaiyangZhou/deep-person-reid
    commit_id=566a56a2cb255f59ba75aa817032621784df546a
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

- deep-person-reid训练默认选用 market1501 数据集，这是一个在行人重识别领域广泛使用的数据集。它包含了来自13个不同场景的1501个行人图像，每个行人有至少一张图像，部分行人有多达五张图像。您可以点击[此链接](https://gitcode.com/Universal-Tool/6378f/?utm_source=article_gitcode_universal&index=bottom&type=card&)从公开网站中下载数据集。

- 请你按以下结构组织 market1501 数据集：

   ```
   ├── market1501
         ├──images
              ├──xxx.jpg
              ├──...
              ├──...
         ├──ori_to_new_im_name.pkl
         ├──partitions.pkl
   ```
### 2.3 准备预训练权重

训练模型需要使用预训练权重osnet_x1_0_imagenet.pth，请提前从[此链接](https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY)下载pth文件，将预训练权重放置在`~/.cache/torch/checkpoints/`路径下。


### 2.4 构建Docker环境

使用Dockerfile，创建运行模型训练所需的Docker环境。

#### 2.4.1 执行以下命令，进入Dockerfile所在目录。

    ```
    cd <modelzoo-dir>/PyTorch/Classification/deep-person-reid
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录。

#### 2.4.2 执行以下命令，构建名为`sdaa_reid`的镜像。

    ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t sdaa_reid
   ```

#### 2.4.3 执行以下命令，启动容器。

    ```
    docker run  -itd --name sdaa_reid -v <dataset_path>:/datasets --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g sdaa_mobilenetv3 /bin/bash
    ```

    其中：`-v`参数用于将主机上的目录或文件挂载到容器内部，对于模型训练，您需要将主机上的数据集目录挂载到docker中的`/datasets/`目录。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。


#### 2.4.4 执行以下命令，进入容器。

    ```
    docker exec -it sdaa_reid /bin/bash
    ```

#### 2.4.5 执行以下命令，启动虚拟环境。

    ```
    conda activate torch_env_py310
    ```

#### 2.4.6 执行以下命令，安装其他环境依赖包。

    ```
    pip install -r requirements.txt
    ```

###  2.4.7 执行以下命令，安装torchreid

    ```
    python setup.py develop
    ```

### 2.5 启动训练
#### 2.5.1 在Docker环境中，进入训练脚本所在目录。
    ```
    cd /workspace/Classification/deep-person-reid
    ```

#### 2.5.2 运行以下命令训练。

  - 检查数据集路径，请参考2.2组织数据集。
  - 使用 Market1501 数据集训练 OSNet：
    ```
    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA
    ```

### 2.6 训练结果

输出训练loss曲线及结果(代码参考[get_loss.py](./get_loss.py))

MeanRelativeError: -0.0028569517
MeanAbsoluteError: -0.01963671
Rule,mean_absolute_error -0.01963671
pass mean_relative_error=-0.0028569517 <= 0.05 or mean_absolute_error=-0.01963671 <= 0.0002

