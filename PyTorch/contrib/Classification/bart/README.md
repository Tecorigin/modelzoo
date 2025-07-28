
# Bart
## 1. 模型概述
BART 是一个序列到序列模型，结合了 BERT 和 GPT 的预训练目标。它通过以不同的方式破坏文本（例如删除单词、打乱句子或屏蔽标记）进行预训练并学习如何修复它。编码器对损坏的文档进行编码，损坏的文本由解码器修复。当它学习恢复原始文本时，BART 在理解和生成语言方面变得非常擅长。


- 论文链接：[1910.13461\]BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension(https://huggingface.co/papers/1910.13461)
- 仓库链接：https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bart.md

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
该数据集为开源数据集，可从 [zhwiki](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2) 下载。

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
    git clone https://gitee.com/xiwei777/tcap_dllogger.git
    cd tcap_dllogger
    python setup.py install
    cd .. 
    pip install -r requirements.txt
    pip3 install numpy==1.24.3
    pip install huggingface_hub
    pip install parameterized
    pip install transformers==4.41.2

    ```

### 2.4 启动训练

1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/bart/run_scripts
    ```

2. 运行训练。该模型支持单机单卡。
    ```
    mkdir -p bart_out && python run_bart.py \
    --train_file ../configs/train_sample.txt \
    --do_train --do_eval \
    --output_dir bart_out \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --max_seq_length 32 \
    --line_by_line   2>&1 | tee sdaa.log
   ```
    更多训练参数参考 run_scripts/argument.py

### 2.5 训练结果
输出训练loss曲线及结果（参考使用[loss.py](./run_scripts/loss.py)）: 

![loss](./run_scripts/loss.jpg)

MeanRelativeError:0.023865286396830804
MeanAbsoluteError:-0.023647474747474748
Rule,mean_absolute_error 0.023865286396830804
pass mean_relative_error=0.023865286396830804 <=0.05 or mean_absolute_error=-0.023647474747474748<=0.0002

