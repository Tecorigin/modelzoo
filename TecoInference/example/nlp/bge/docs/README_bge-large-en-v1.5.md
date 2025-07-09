# Bge-large-en-v1.5

## 1. 模型概述

Bge-large-en-v1.5 是一个高性能的英文嵌入模型，专门用于文本相似度和信息检索任务。该模型是 BGE（BAAI Embedding）模型系列的一部分。它在多个基准测试中表现出色，能够有效捕捉文本语义并为自然语言处理（NLP）应用提供高质量的特征表示。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 环境准备：介绍推理前需要完成的基础环境和第三方依赖的检查和安装。
2. 获取ONNX文件：介绍如何获取推理所需的ONNX模型文件。
3. 获取数据集：介绍如何获取推理所需的数据集。
4. 启动推理：介绍如何运行推理。
5. 精度验证：介绍如何验证推理精度。

### 2.1  环境准备

请参考[基础环境安装](../README.md)章节，完成推理前的环境依赖检查和安装。

### 2.2 获取ONNX文件

1. 下载[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)模型权重到`BAAI/bge-large-en-v1.5`目录。

2. 执行以下命令，导出模型的ONNX文件。

    ```
    mkdir onnx_fles
    optimum-cli export onnx \
    --model BAAI/bge-large-en-v1.5 onnx_fles/bge-large-en-v1.5 \
    --task feature-extraction \
    --framework pt \
    --batch_size 1 \
    --sequence_length 512 \
    --monolith \
    --optimize O1  
    ```

3. 执行如下脚本进行简化。

    ```
    python simplify.py --onnx onnx_fles/bge-large-en-v1.5/model.onnx --save_name bge-large-en-v1.5.onnx --need_fp16 True
    ```

### 2.3 获取数据集

您可以通过以下方式获取推理所需的数据集：

- 使用内置的demo数据样本。Demo数据样本位于仓库的`./datas`目录。
- 使用mteb/nfcorpus数据集，从官方[mteb/nfcorpus](https://huggingface.co/datasets/mteb/nfcorpus)下载。

### 2.4 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

    ```
    cd modelzoo/TecoInference/example/nlp/bge
    ```

2. 运行推理。

    **注意**: 推理需要模型配置文件, 请参考[获取onnx文件](#22-获取onnx文件)小节下载包含配置文件的模型文件夹。

    单样本推理：使用单个样本作为输入，进行推理。

     ```
     python example_single_batch.py \
        --ckpt bge-large-en-v1.5_dyn_fp16.onnx \
        --config-path BAAI/bge-large-en-v1.5 \
        --data-path datas/bge-large-en_demo.txt \
        --model_name bge-large-en-v1.5 \
        --batch-size 2 \
        --shape 512 \
        --half \
        --target sdaa
    ```

    推理结果：

    ```
    ["sample-data-1", "sample-data-2"]-cosine_sim:1.0
    ```


模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data-path    | 数据路径 |N/A|
| config-path  | config路径 |N/A|
| ckpt         | 模型onnx路径  | N/A |
| batch-size   | 推理的batch_size  | 1 |
| shape        | 模型的shape  | 512 |
| target       | 推理的设备 | `sdaa` |
| half         | 模型推理是否使用`float16`  | True |


### 2.5 精度验证

请提前准备数据集，执行以下命令，获得推理精度数据。
```
python example_valid.py --model_name bge-large-en-v1.5 --ckpt bge-large-en-v1.5_dyn_fp16.onnx --config-path BAAI/bge-large-en-v1.5 --data-path mteb/nfcorpus/ --batch-size 128 --half --target sdaa
```

精度结果如下：

```
summary: avg_sps: 92.37054503698127, e2e_time: 44.34313988685608, data_time: 0.013465166091918945, avg_inference_time: 0.7184663446249999, avg_preprocess_time: 0.37218762189149857, avg_postprocess: 0.053388483822345734
eval_metric: 0.36843
```

#### 精度结果说明

 精度结果说明如下：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| eval_metric      | 数据集验证精度  |