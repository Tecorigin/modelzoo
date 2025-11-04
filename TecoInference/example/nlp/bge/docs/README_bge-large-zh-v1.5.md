# Bge-large-zh-v1.5

## 1. 模型概述

Bge-large-zh-v1.5是由北京智源人工智能研究院开发的中文通用嵌入模型，属于BGE模型系列。它基于Transformer架构，专为中文场景设计，具有强大的中文处理能力。

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

1. 下载[bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)模型权重到`BAAI/bge-large-zh-v1.5`目录。

2. 执行以下命令，导出模型的ONNX文件。

    ```
    mkdir onnx_fles
    optimum-cli export onnx \
        --model BAAI/bge-large-zh-v1.5 onnx_fles/bge-large-zh-v1.5 \
        --task feature-extraction \
        --framework pt \
        --batch_size 1 \
        --sequence_length 512 \
        --monolith \
        --optimize O1 
    ```

3. 执行如下脚本进行简化。

    ```
    python simplify.py --onnx onnx_fles/bge-large-zh-v1.5/model.onnx --save_name bge-large-zh-v1.5.onnx --need_fp16 True
    ```

### 2.3 获取数据集

您可以通过以下方式获取推理所需的数据集：

- 使用内置的demo数据样本。Demo数据样本位于仓库的`./datas`目录。
- 使用C-MTEB/OCNLI数据集，从官方[C-MTEB/OCNLI](https://huggingface.co/datasets/C-MTEB/OCNLI)下载。

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
        --ckpt bge-large-zh-v1.5_dyn_fp16.onnx \
        --config-path BAAI/bge-large-zh-v1.5 \
        --data-path datas/bge-large-zh_demo.txt \
        --model_name bge-large-zh-v1.5 \
        --batch-size 2 \
        --shape 512 \
        --half \
        --target sdaa
    ```

    推理结果：

    ```
    ["样例数据-1", "样例数据-2"]-cosine_sim:1.0
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
python example_valid.py --model_name bge-large-zh-v1.5 --ckpt bge-large-zh-v1.5_dyn_fp16.onnx --config-path BAAI/bge-large-zh-v1.5 --data-path C-MTEB/OCNLI/ --batch-size 128 --half --target sdaa
```

精度结果如下：

```
summary: avg_sps: 121.11961304864882, e2e_time: 28.53377676010132, data_time: 0.0171201229095459, avg_inference_time: 0.7184411683703703, avg_preprocess_time: 0.10564990396852847, avg_postprocess: 0.17738782034979927
eval_metric: 0.779392973534058
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