#  Hashnerf
## 1. 模型概述
HashNeRF是一种创新的神经辐射场（Neural Radiance Fields, NeRF）变体，旨在提高渲染速度和存储效率。传统的NeRF模型在表示复杂场景时需要大量的计算资源和时间来训练以及渲染新视图，这是因为它们依赖于通过多层感知机（MLP）直接从连续坐标映射到色彩和密度值。HashNeRF采用了一种不同的方法，它利用可微分的哈希表来存储场景的特征表示，这些哈希表在训练过程中被优化。这种方法允许模型以更紧凑的形式存储信息，并且能够显著加快查询速度，从而实现更快的渲染性能。
具体来说，HashNeRF将输入的空间位置和视角方向编码为一个高维向量，并通过查找预先分配的哈希表来检索相应的特征。这些特征随后被馈送到一个小的MLP中，用于预测该点的颜色和体积密度。由于使用了哈希结构，使得HashNeRF可以在不牺牲图像质量的前提下，极大地减少内存占用并缩短渲染时间。此外，这种架构上的改进也使得处理大规模场景成为可能，增强了NeRF技术在实际应用中的可行性。因此，HashNeRF不仅继承了NeRF的优点，还解决了其主要局限性，展示了在3D场景重建与逼真渲染领域的巨大潜力

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装
请参考[基础环境安装](../../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集
Hashnerf数据集下载运行load_data_by_kaggle.py文件
(该模型数据集与nerf一致)

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
将数据集拷贝到指定目录下：/HashNeRF/data/
数据集目录结构：
```angular2html
    |----data
        |------hotdog
            |-----test
            |-----train
            |-----val
            |----tranforms_train.json
            |----tranforms_test.json
            |----tranforms_val.json
```

### 2.3 构建环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
   
2. 所需安装的包
    ```
    pip install -r requirements.txt
    ```
    
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/3D/HashNeRF
    ```
   
2. 运行训练。

   ```
   python run_nerf.py \
    --config configs/hotdog.txt \
    --finest_res 512 \
    --log2_hashmap_size 19 \
    --lrate 0.01 \
    --lrate_decay 10
   ```


### 2.5 训练结果
模型训练2h，得到结果如下  
|HashNeRF|Epoch|Loss|PSNR| 
| :-: | :-: | :-: | :-: | 
|HashNeRF|1|0.00367|29.1719|

注：由于模型计算的hash值会溢出变成nan，所以模型fallback到cpu上运行