# bge-reranker-large

## 1. 模型概述

Bge-reranker-large是一个基于大型语言模型的重排工具，它专门设计用于改善搜索结果或推荐系统的输出质量。这种工具通常使用机器学习技术，根据用户的行为或偏好，对已有的搜索结果或推荐列表进行重新排序，以提供更加个性化和相关的结果。

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

1. 从[官方](https://huggingface.co/BAAI/bge-reranker-large)下载模型权重。

2. 执行以下命令，导出模型的ONNX文件。

    ```
    optimum-cli export onnx \
        --model <path_to_bge-reranker-large_file> <onnx_save_path> \
        --task text-classification \
        --framework pt \
        --batch_size <bs> \
        --sequence_length 512 \
        --monolith \
        --optimize O1 
    ```
    其中：
    - `<path_to_bge-reranker-large_file>`为bge-reranker-large文件夹路径。
    - `<onnx_save_path>`为保存的文件夹地址。
    - `<bs>`为batch size。


3. 执行如下脚本进行简化。

    ```
    python simplify.py --onnx ./bge-reranker-large/model.onnx --save_name bge-reranker-large.onnx --save_as_external_data True
    ```

### 2.3 获取数据集

您可以通过以下方式获取推理所需的数据集：

首先从[百度网盘](https://pan.baidu.com/s/16E4TLaFB5ZJbYs57xp9gPQ?pwd=pu89)下载`config`文件：
- 使用内置的demo数据样本。Demo数据样本位于仓库的`./datas`目录。
- 使用mrpc数据集，[百度网盘](https://pan.baidu.com/s/16E4TLaFB5ZJbYs57xp9gPQ?pwd=pu89)下载`mrpc.jsonl`数据集文件和`datas`onnx精度数据文件。

### 2.4 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

    ```
    cd modelzoo/TecoInference/example/nlp/bge
    ```

2. 运行推理。

    - 单样本推理：使用单个样本作为输入，进行推理

        ```
        python example_single_batch.py \
                --ckpt bge-reranker-large.onnx \
                --config-path path_to/config \
                --data-path datas/bge-reranker-large_demo.txt \
                --model_name bge-reranker-large \
                --batch-size 2 \
                --shape 512 \
                --half \
                --target sdaa
        ```

        推理结果：

        ```
        [['what is panda?', 'hi']]-相似度:-5.609375
        [['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]-相似度:5.76171875
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
    --ckpt bge-reranker-large.onnx \
    --config-path path_to/config \
    --onnx-datas path_to/datas \
    --data-path path_to/mrpc.jsonl \
    --model_name bge-reranker-large \
    --batch-size 4 \
    --shape 512 \
    --half \
    --target sdaa
```

精度结果如下：

```
eval_metric:0.0011218713214037903
summary: avg_sps: 64.30263708569609 images/s, e2e_time: 30.300684452056885 s, avg_inference_time: 0.07014633230079811 s, avg_preprocess_time: 1.8643560232939542e-05 s, avg_postprocess: 1.0574857393900553e-05 s
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
