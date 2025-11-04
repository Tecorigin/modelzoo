# Transformer

## 1.模型概述

Transformer是一种使用自注意力机制处理数据序列的模型架构，广泛应用于机器翻译、语言建模等序列任务中。来自于[fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)，使用IWSLT14数据进行训练，训练环境为`fairseq`。

## 2.快速开始

### 2.1环境配置

请参考[基础环境安装](https://gitee.com/tecorigin/modelzoo/blob/main/doc/Environment.md)章节，完成训练前的基础环境检查和安装。

```bash
pip install -r requirements.txt
pip install -e .
```

### 2.2数据集获取与准备

使用以下命令获取数据集并进行预处理。
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

### 2.3模型训练

使用以下命令训练模型。
```bash
bash examples/translation/train_transformer.sh
```

### 2.4结果展示

训练loss曲线如下图所示。

![loss figure](loss.png)

可以看出随着训练的进行，loss整体呈现下降趋势。
