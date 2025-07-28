
# Electra
## 1. 模型概述
ELECTRA修改了 BERT 等传统掩码语言模型的预训练目标。ELECTRA 不再仅仅掩码标记并要求模型预测它们，而是训练两个模型：一个生成器和一个鉴别器。生成器用合理的替代词替换部分标记，而鉴别器（您实际使用的模型）则学习检测哪些标记是原始标记，哪些是被替换的。这种训练方法非常高效，并且可以扩展到更大的模型，同时显著减少计算量。

这种方法非常高效，因为 ELECTRA 会从输入中的每个 token 中进行学习，而不仅仅是那些被屏蔽的 token。正因如此，即使是小型 ELECTRA 模型也能在消耗更少计算资源的情况下，达到甚至超越大型模型的性能。


- 论文链接：[2003.10555\]ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators(https://huggingface.co/papers/2003.10555)
- 仓库链接：https://github.com/huggingface/transformers/blob/main/i18n/README_zh-hans.md

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
Deit 使用 ImageNet 数据集，该数据集为开源数据集，可从 [zhwiki](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2) 下载。

#### 2.2.2 处理数据集
具体配置方式可参考：https://blog.csdn.net/weixin_39709674/article/details/111847635。


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    cd .. 
    pip install -r requirements.txt
    pip3 install numpy==1.24.3
    pip install huggingface_hub==0.33.4
    pip install parameterized
    pip install transformers==4.41.2
    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/electra/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
    mkdir -p electra_out &&  python run_electra.py   --model_name_or_path ../configs/electra-small-generator   \
    --train_file ../configs/train.txt   \
    --do_train --do_eval   \
    --output_dir electra_out   \
    --overwrite_output_dir   \
    --per_device_train_batch_size 16   \
    --max_seq_length 32   \
    --line_by_line
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./run_scripts/loss.jpg)

MeanRelativeError:-0.01308239642928053
MeanAbsoluteError:-0.14012299999999991
Rule,mean_absolute_error -0.01308239642928053
passmean_relative_error=-0.01308239642928053 <=0.05 or mean_absolute_error=-0.14012299999999991<=0.0002

