# Bge-large-zh

## 1. 模型概述

Bge-large-zh是一个大型的中文语言模型，专为处理和理解中文文本而设计。它能够执行多种语言任务，如文本生成、翻译、摘要、问答等，通过深度学习技术，Bge-large-zh能够提供准确、自然的语言处理能力，以满足用户在不同场景下的语言需求。

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

1. 从[官方](https://hf-mirror.com/BAAI/bge-large-zh/tree/main)下载模型权重到当前目录。

2. 执行以下命令，导出模型的ONNX文件。

    ```
    optimum-cli export onnx \
        --model <path_to_bge-large-zh_file> <onnx_save_path> \
        --task feature-extraction \
        --framework pt \
        --batch_size <bs> \
        --sequence_length 512 \
        --monolith \
        --optimize O1 
    ```
    其中：
    - `<path_to_bge-large-zh_file>`：为bge-large-zh文件夹路径。
    - `<onnx_save_path>`：为保存的文件夹地址。
    - `<bs>`：为batch size。


3. 执行如下脚本进行简化。

    ```
    python simplify.py --onnx ./bge-large-zh/model.onnx --save_name bge-large-zh.onnx
    ```

### 2.3 获取数据集

您可以通过以下方式获取推理所需的数据集：

**注意**：首先从[百度网盘](https://pan.baidu.com/s/1DYTim1lm2pTOkRyk10LZDw?pwd=qar6)下载`config`文件。
- 使用内置的demo数据样本。Demo数据样本位于仓库的`./datas`目录。
- 使用C-MTEBOCNL数据集，[百度网盘](https://pan.baidu.com/s/1DYTim1lm2pTOkRyk10LZDw?pwd=qar6)下载`C-MTEBOCNL_validation.jsonl`数据集文件和`datas`onnx精度数据文件。

### 2.4 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

    ```
    cd modelzoo/TecoInference/example/nlp/bge
    ```

2. 运行推理。

    单样本推理：使用单个样本作为输入，进行推理。

     ```
     python example_single_batch.py \
             --ckpt bge-large-zh.onnx \
             --config-path path_to/config \
             --data-path datas/bge-large-zh_demo.txt \
             --model_name bge-large-zh \
             --batch-size 2 \
             --shape 512 \
             --half \
             --target sdaa
    ```

    推理结果：

    ```
    ["样例数据-1", "样例数据-2"]-cosine_sim:0.9652644395828247
    ```


模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data-path    | 数据路径 |datas/bge-large-zh_demo.txt|
| config-path  | config路径 |N/A|
| ckpt         | 模型onnx路径  | N/A |
| batch-size   | 推理的batch_size  | 1 |
| shape        | 模型的shape  | 512 |
| target       | 推理的设备 | `sdaa` |
| half         | 模型推理是否使用`float16`  | True |


### 2.5 精度验证

请提前准备数据集，执行以下命令，获得推理精度数据。
```
python example_valid.py \
    --ckpt bge-large-zh.onnx \
    --config-path path_to/config \
    --onnx-datas path_to/datas \
    --data-path path_to/C-MTEBOCNL_validation.jsonl \
    --model_name bge-large-zh \
    --batch-size 4 \
    --shape 512 \
    --half \
    --target sdaa
```

精度结果如下：

```
eval_metric:0.0005093644852680355
summary: avg_sps: 62.99533587490068 images/s, e2e_time: 67.0028817653656 s, avg_inference_time: 0.07109454251477197 s, avg_preprocess_time: 3.5313816813679484e-05 s, avg_postprocess: 0.0014127650302209896 s
```
 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| eval_metric      | 数据集验证精度  |