#  RNN-T
## 1. 模型概述
RNN-Transducer 是一种端到端的语音识别模型，融合了循环神经网络（RNN）与变换器（Transducer）结构。它主要由三个部分组成：编码器、预测网络和解码器。编码器负责将输入的语音信号转换为一系列的上下文向量，捕捉语音中的时间序列特征；预测网络基于之前已生成的输出，预测下一个可能的字符或标签；解码器则通过结合编码器输出的上下文向量和预测网络的预测结果，生成最终的识别结果。​与传统的语音识别模型相比，RNN-Transducer 无需进行音素分割和语言模型重打分等复杂的中间步骤，能够直接从语音信号映射到文本输出，大大简化了语音识别流程。这种端到端的架构有效解决了传统模型中各模块之间的不一致性问题，提高了识别的准确性和效率。同时，RNN-Transducer 能够处理不同长度的输入序列，对语音中的连读、语速变化等现象具有较好的鲁棒性。
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
LJSpeech-1数据集下载地址如下
链接: https://keithito.com/LJ-Speech-Dataset/ 

#### 2.2.2 处理数据集
解压数据集，将其按格式放入文件datasets中，格式目录如工作目录所示

#### LJSpeech数据集和权重目录结构：
```angular2html
|-- LJSpeech
    |-- metadata.csv 
    |-- wavs               
        |-- LJ015-0267.wav      
        |-- LJ010-0307.wav      
        |-- LJ010-0300.wav  
        |--........        
```
#### 数据处理：
    ```
    cd scripts
    ```
   

    ```
    python3 process_data.py
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
    cd <ModelZoo_path>/PyTorch/contrib/speech/RNNT
    ```
   
2. 运行训练。


   ```
   torchrun --nproc_per_node 4 train.py
   ```

### 2.5 训练结果

**训练loss曲线:**

![train_loss.png](script%2Ftrain_loss.png)

**模型测试准确率：**

![train_acc.png](script%2Ftrain_acc.png)
