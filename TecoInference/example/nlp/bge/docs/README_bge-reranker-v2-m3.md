# Bge-reranker-v2-m3

## 1. 模型概述

BGE-reranker-v2-m3 是一款由北京智源人工智能研究院开发的轻量级多语言重排序模型。它基于 BGE-M3 架构优化，支持100多种语言，尤其在中英文混合检索中表现出色。该模型采用分层自蒸馏策略，推理速度更快，适合高并发场景。它还支持“文本+图片”混合检索，可与 BGE-M3 的稠密/稀疏检索无缝结合，广泛应用于搜索引擎、问答系统和推荐系统等场景。

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

1. 下载[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)模型权重到`BAAI/bge-reranker-v2-m3`目录。

2. 执行以下命令，导出模型的ONNX文件。

    ```
    mkdir onnx_fles
    optimum-cli export onnx \
    --model BAAI/bge-reranker-v2-m3 onnx_fles/bge-reranker-v2-m3 \
    --task text-classification \
    --framework pt \
    --batch_size 1 \
    --sequence_length 8192 \
    --monolith \
    --optimize O1 
    ```

3. 执行如下脚本进行简化。

    ```
    python simplify.py --onnx onnx_fles/bge-reranker-v2-m3/model.onnx --save_name bge-reranker-v2-m3_dyn_fp32.onnx --save_as_external_data True
    ```

### 2.3 获取数据集

您可以通过以下方式获取推理所需的数据集：

- 使用内置的demo数据样本。Demo数据样本位于仓库的`./datas`目录。
- 使用mldr/en数据集，从[百度网盘](https://pan.baidu.com/s/1gzG-d9WVzaK32bfeDCBntQ?pwd=gpkn)下载。

### 2.4 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

    ```
    cd modelzoo/TecoInference/example/nlp/bge
    ```

2. 运行推理。

    **注意**: 推理需要模型配置文件, 请参考[获取onnx文件](#22-获取onnx文件)小节下载包含配置文件的模型文件夹。

    单样本推理：使用单个样本作为输入，进行推理。

     ```
     python example_single_batch.py --model_name bge-reranker-v2-m3 \
        --ckpt bge-reranker-v2-m3_dyn_fp32.onnx \
        --config-path BAAI/bge-reranker-v2-m3 \
        --data-path datas/bge-reranker-large_demo.txt \
        --batch-size 2 \
        --shape 512 --half --target sdaa
    ```

    推理结果：

    ```
    [['what is panda?', 'hi']]-相似度:-8.1796875
    [['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]-相似度:5.265625
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
python example_valid.py --model_name bge-reranker-v2-m3 --ckpt bge-reranker-v2-m3_dyn_fp32.onnx --config-path BAAI/bge-reranker-v2-m3 --data-path mldr/data --half --batch-size 128 --target sdaa
```

精度结果如下：

```
summary: avg_sps: 27.651312324189515, e2e_time: 2893.171906709671, data_time: 0.13800334930419922, avg_inference_time: 0.7200764488064003, avg_preprocess_time: 3.907718627166748, avg_postprocess: 0.00027131729125976563
mldr evaluation on ['en'] completed.
Start computing metrics.
eval_metric: 0.3882
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