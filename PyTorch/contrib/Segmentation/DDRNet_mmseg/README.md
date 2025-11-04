# DDRNet

## 1. 模型概述

DDRNet 即 Deep Dual-Resolution Networks，是一种用于语义分割的神经网络模型。它创新地采用了深度双分辨率网络结构，通过并行的高低分辨率分支来提取特征，低分辨率分支获取丰富的上下文信息，高分辨率分支保留精细的空间细节，二者在不同阶段进行融合，平衡了速度与精度。DDRNet 在实时语义分割任务中表现突出，像自动驾驶的场景分割、智能安防的目标分割等领域都有广泛应用，凭借其独特的双分辨率设计和高效的特征融合机制，成为深度学习语义分割领域的重要模型之一，为解决实际场景中的语义分割问题提供了有力支持。

## 2. 快速开始

### 2.1 基础环境安装

请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。


### 2.2 数据集准备
#### 2.2.1 数据集准备

我们在本项目中使用了 Cityscapes 数据集。链接：https://www.cityscapes-dataset.com/

#### 2.2.2 处理数据集

- 解压训练数据集：
- 现在/data/datasets/cityscapes/中包含已经转换好的数据集，因此可以直接使用，无需运行下述命令。
``` bash
cd /data/datasets/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
python -m pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py /data/datasets/cityscapes/  
```

#### 2.2.3 数据集目录结构

数据集目录结构参考如下所示:

```
Cityscapes/
|-- gtFine
|   |-- test
|   |-- train
|   `-- val
`-- leftImg8bit
    |-- test
    |-- train
    `-- val
```


### 2.3 构建环境
所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动环境。
参考[官方教程](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/zh_cn/get_started.md)进行安装
```sh
conda create -n mmseg--clone torch_env
conda activate mmseg
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
- 若在安装mmengine时报错："GLIBCXX not found"
参考以下[博客](https://zhuanlan.zhihu.com/p/685165815)和代码解决：
```sh
cd /root/anaconda3/envs/mmseg/lib/ 
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  .
ln -s -f libstdc++.so.6.0.30 libstdc++.so.6
```
- 注意： mmseg库需要修改site-package中的mmengine环境依赖。建议直接clone迁移好的mmseg环境。

2. 下载代码
``` 
git clone https://github.geekery.cn/https://github.com/open-mmlab/mmsegmentation.git
```
3. 安装依赖
```
pip install -v -e .
pip install mmcv==2.1.0
pip install ftfy
pip install regex
```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/Segmentation/DDRNet_mmseg
    ```

2. 运行训练。
    ```
    export SDAA_VISIBLE_DEVICES=0,1,2,3
    timeout 120m sh tools/dist_train.sh configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024.py 4 | tee ./scripts/train_sdaa_3rd.log
    ```

### 2.5 训练结果

- 可视化命令
    ```
    cd ./script
    python plot_curve.py
    ```
 2h训练了3700个iter
| 加速卡数量 | 模型 | 混合精度 | Batch Size | iter | train_loss | AccTop1 |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | DDRNet | 否 | 2 | 850 | 6.2965 | - | 