# BERT finetune

## 1. 模型概述

BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的语言模型，它在大规模的无监督数据上进行了训练，学习到了丰富的语言表征，拥有强大的自然语言理解能力（Natural Language Understanding, NLU）。对于NLP子任务，可使用标注数据集在BERT预训练模型的基础上进行有监督训练，使BERT模型进一步理解任务，从而对子任务有更好的效果。

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 构建Docker环境：介绍如何使用Dockerfile创建模型训练时所需的Docker环境。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 构建Docker环境

1. 执行以下命令，进入Dockerfile所在目录。
    ```
    cd <modelzoo-dir>/PyTorch/NLP/BERT_finetune
    ```
    其中： `modelzoo-dir`是ModelZoo仓库的主目录

2. 执行以下命令，构建名为`bert_pt`的镜像。
   ```
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t  bert_pt
   ```

3. 执行以下命令，启动容器。
   ```
   docker run  -itd --name bert_pt --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g bert_pt /bin/bash
   ```

   其中：如果物理机上有数据集和权重，请添加`-v`参数用于将主机上的目录或文件挂载到容器内部，例如`-v <host_data_path>/<docker_data_path>`。更多容器配置参数说明参考[文档](../../../doc/Docker.md)。

4. 执行以下命令，进入容器。
    ```
   docker exec -it bert_pt /bin/bash
   ```
5. 执行以下命令，启动虚拟环境。
   ```
   conda activate torch_env
   ```

### 2.3 启动训练

- 进入训练脚本所在目录
   ```
   cd /workspace/NLP/BERT_finetune
   ```

- 运行训练

该模型对问答、文本摘要、情感分析、文本分类四个任务进行支持。

#### 2.3.1 问答任务-使用SQuAD v1.1数据集

1. 数据集获取

    ``` bash
    cd /workspace/NLP/BERT_finetune/data/squad
    # 下载完成后，会保存至/workspace/NLP/BERT_finetune/data/squad/v1.1
    sh squad_download.sh
    ```

2. 预训练权重获取

    本项目使用Nvidia提供的BERT-base-uncased预训练权重进行训练。可前往[NVIDIA-NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_pretraining_amp_lamb/files)下载。


3. 运行训练。该模型支持单机单卡、单机八卡。

    - 单机单卡
    ``` bash
    cd /workspace/NLP/BERT_finetune/
    # 通过下载获取数据集，请设置 --dataset_path data/squad/v1.1，/ckpt/bert_base.pt为下载的权重路径
    python run_scripts/run_bert_base_squad_v1.1.py --model_name bert_base_uncased --nproc_per_node 4 --bs 4 --lr 3e-5 --device sdaa --epoch 5 --step -1 --dataset_path data/squad/v1.1 --grad_scale True --autocast True --checkpoint_path /ckpt/bert_base.pt --warm_up 0.1 --max_seq_length 384 --do_predict True
    ```
    - 单机八卡

    ``` bash
    # 通过下载获取数据集，请设置 --dataset_path data/squad/v1.1，/ckpt/bert_base.pt为下载的权重路径
    python run_scripts/run_bert_base_squad_v1.1.py --model_name bert_base_uncased --nproc_per_node 32 --bs 4 --lr 3e-5 --device sdaa --epoch 5 --step -1 --dataset_path data/squad/v1.1 --grad_scale True --autocast True --checkpoint_path /ckpt/bert_base.pt --warm_up 0.1 --max_seq_length 384 --do_predict True
    ```

4. 训练结果

    |加速卡数量 | Epochs | 混合精度 |Batch size|max seq len|  Acc| extra_match|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |1| 5|是|16|384| 89.01% | 82.10 |
    |8| 5|是|64|384| 88.61% | 81.47 |

#### 2.3.2 文本摘要任务-使用CNN/DM数据集

1. 数据集获取

    本任务的数据集来自[BertSum项目仓库](https://github.com/nlpyang/BertSum)，仓库中提供了[处理后的数据](https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6)，可直接进行训练。


2. rouge安装

    为了计算文本生成指标rouge，需要进行如下操作：

    ``` bash
    git clone https://github.com/andersjo/pyrouge.git rouge

    # 注意这里需要配置绝对路径
    pyrouge_set_rouge_path path/to/rouge/tools/ROUGE-1.5.5/

    cd rouge/tools/ROUGE-1.5.5/data
    rm WordNet-2.0.exc.db
    ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
    ```

3. 运行训练。该模型支持单机单卡。

    - 单机单卡
    ``` bash
    cd /workspace/NLP/BERT_finetune/
    # <path>/cnndm_data为下载的数据集路径，/ckpt/bert_base.pt为下载的权重路径
    python run_scripts/run_bert_base_cnndm.py --model_name bert_base_uncased --nproc_per_node 4 --bs 4 --lr 2e-3 --device sdaa --step 20000 --dataset_path <path>/cnndm_data --grad_scale True --autocast True --checkpoint_path /ckpt/bert_base.pt --warm_up 0.2 --max_seq_length 512 --do_predict True
    ```

4. 训练结果

    |加速卡数量 | steps | 混合精度 |Batch size|max seq len|  ROUGE-1| ROUGE-2|ROUGE-L|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |1| 20000 |是|16|512|  42.94 | 20.07 |39.40|


#### 2.3.3 情感分析任务-使用SST2数据集

1. 数据集获取

    SST2一个情感分析数据集，主要针对电影评论来做情感分类，因此SST属于单个句子的文本分类任务。huggingface官方提供了[原始数据集](https://huggingface.co/datasets/gpt3mix/sst2/tree/main/data)下载。


2. 运行训练。该模型支持单机单卡、单机八卡。

    - 单机单卡
    ``` bash
    cd /workspace/NLP/BERT_finetune/
    # <path>/sst2为下载的数据集路径，/ckpt/bert_base.pt为下载的权重路径
    python run_scripts/run_bert_base_imdb.py --model_name bert-base-uncased --nproc_per_node 4 --bs 4 --lr 2.4e-5 --device sdaa --epoch 5 --step -1 --dataset_path <path>/sst2 --checkpoint_path /ckpt/bert_base.pt --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True
    ```
    - 单机八卡
    ``` bash
    # <path>/sst2为下载的数据集路径，/ckpt/bert_base.pt为下载的权重路径
    python run_scripts/run_bert_base_imdb.py --model_name bert-base-uncased --nproc_per_node 32 --bs 4 --lr 2.4e-5 --device sdaa --epoch 5 --step -1 --dataset_path <path>/sst2 --checkpoint_path /ckpt/bert_base.pt --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True
    ```

3. 训练结果

    |加速卡数量  | Epochs | 混合精度 |Batch size|max seq len| 吞吐量|acc|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |1| 5 |是|16|128| - | 85.56% |
    |8| 5 |是|64|128| - | 88.56% |

#### 2.3.4 文本分类-使用Thunews数据集

1. 数据集获取

    tnews是一个多分类的中文新闻数据集，包括 财经, 彩票, 房产, 股票, 家居, 教育, 科技, 社会, 时尚, 时政, 体育, 星座, 游戏, 娱乐 14种新闻。默认使用混合精度训练，可以用两种方法获取训练使用的数据集。清华官方提供了[Thunews数据集](http://thuctc.thunlp.org/)下载。

    ``` bash
    # 下载完成后进行解压
    mkdir thunews_dataset && unzip THUCNews.zip -d thunews_dataset

    cd /workspace/NLP/BERT_finetune/data
    # 修改tnews_dir与output_dir，tnews_dir为数据集解压路径，output_dir为处理完后的数据保存路径
    vim process_tnews.py

    # 执行完成后output_dir中会生成train.tsv dev.tsv两个文件
    python process_tnews.py
    ```

2. 预训练权重获取
中文BERT权重可前往[ModelScope](https://www.modelscope.cn/models/dienstag/chinese-bert-wwm/files)下载。

3. 运行训练。该模型支持单机单卡。
- 单机单卡
    ``` bash
    cd /workspace/NLP/BERT_finetune/
    # /dataset/tnews为下载的数据集路径，/ckpt/pytorch_model.bin为下载的权重路径
    python run_scripts/run_bert_base_thucnews.py --model_name bert-base-chinese --nproc_per_node 4 --bs 16 --lr 2.4e-5 --device sdaa --epoch 4 --dataset_path /dataset/tnews --checkpoint_path /ckpt/pytorch_model.bin --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True
    ```
    训练命令参数说明参考[文档](run_scripts/README.md)。
