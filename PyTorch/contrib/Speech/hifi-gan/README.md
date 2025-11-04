#  HiFi-GAN
## 1. 模型概述
HiFi-GAN（High Fidelity Generative Adversarial Network）是一种基于生成对抗网络（GAN）架构的高效语音合成声码器模型。它由生成器和判别器组成，生成器通过多尺度残差块和上采样模块，从低维梅尔频谱中生成高保真的波形音频；判别器则负责区分生成的音频和真实音频，以促使生成器不断优化。
HiFi-GAN 解决了传统声码器计算成本高、合成音频质量有限的问题，能够在保持低推理复杂度的同时，合成具有高分辨率和丰富细节的语音。该模型生成的语音在音质、自然度和韵律表现上都达到了极高的水平，有效提升了语音合成的整体效果。
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
LJSpeech数据集下载地址如下
链接: https://keithito.com/LJ-Speech-Dataset/ 

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，将下载的wavs文件复制到/hifi-gan/LJSpeech-1.1 中。
#### LJSpeech数据集结构：
文件目录结构：
```angular2html
  |--LJSpeech-1.1
        |-- wavs（从datasets中获取）
        |-- training.txt(已有)
        |-- validation.txt(已有)


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
    cd <ModelZoo_path>/PyTorch/contrib/Speech/hifi-gan
    ```
   
2. 运行训练。

   ```
   torchrun --nproc_per_node 4 train.py --config config_v1.json
   ```

### 2.5 训练结果
模型训练4h，得到结果如下  
|加速卡数量|模型|Epoch|Batch size|Loss|  
| :-: | :-: | :-: | :-: | :-: | :-: |   
|4|hifi-gan|15|32|90|82.878|

