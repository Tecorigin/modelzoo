# Jina-embeddings-v2-base-code

## 1. 模型概述

Jina-embeddings-v2-base-code 是由 Jina AI 提供的一个深度学习模型，专门用于生成文本嵌入向量。该模型基于 Transformer 架构，经过大量文本数据的预训练，能够将文本片段（如单词、句子或段落）映射到高维空间中的稠密向量。这些向量保留了原始文本的语义信息，可用于多种自然语言处理任务，如语义搜索、文本分类和聚类等。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取ONNX文件：介绍如何获取推理所需的ONNX模型文件。
4. 获取数据集：介绍如何获取推理所需的数据集。
5. 启动推理：介绍如何运行推理。
6. 精度验证：介绍如何验证推理精度。

### 2.1 基础环境安装

请参考推理首页的[基础环境安装](../../../README.md)章节，完成推理前的基础环境检查和安装。

### 2.2 安装第三方依赖

1. 执行以下命令，进入容器。

   ```shell
   docker exec -it model_infer bash
   ```

2. 执行以下命令，进入conda环境。
   ```shell
   conda activate tvm-build_py310
   ```

3. 执行以下命令，进入第三方依赖安装脚本所在目录。

   ```shell
   cd <modelzoo_dir>/TecoInference/example/nlp/jina
   ```

4. 执行以下命令，安装第三方依赖。

   ```shell
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取ONNX文件

1. 下载模型权重到当前路径。

   - [jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)

2. 执行以下命令，导出模型的ONNX文件，使用生成的`jina-embeddings-v2-base-code_dyn_fp16.onnx`进行推理。

   ```bash
   mkidr onnx_fles
   optimum-cli export onnx \
    --model jinaai/jina-embeddings-v2-base-code onnx_fles/jina-embeddings-v2-base-code \
    --trust-remote-code \
    --task feature-extraction \
    --opset 17 \
    --framework pt \
    --batch_size 1 \
    --sequence_length 8192 \
    --monolith \
    --optimize O1 

   python simplify.py --onnx onnx_fles/jina-embeddings-v2-base-code/model.onnx --save_name jina-embeddings-v2-base-code.onnx
   ```
   参数说明：
   | 参数 | 说明 |
   | ------------- | ------------- |
   | onnx | 模型的权重文件|
   | save_name | 保存的模型名称/路径 |

### 2.4 获取数据集

通过以下方式获取推理所需的数据集：
- 使用内置的demo数据样本。Demo数据样本位于仓库的`./demo`目录。
- 使用code-search-net数据集用于模型推理和推理精度验证，参考链接[code_search_net](https://huggingface.co/datasets/code-search-net/code_search_net)下载。

### 2.5 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

   ```bash
   cd <modelzoo_dir>/TecoInference/example/nlp/jina
   ```

2. 运行推理。

    **注意**: 推理需要模型配置文件, 请参考[获取onnx文件](#23-获取onnx文件)小节下载包含配置文件的模型文件夹。

    - 单样本推理：

        ```bash
        python example_single_batch.py --model_name jina-embeddings-v2-base-code --ckpt jina-embeddings-v2-base-code_dyn_fp16.onnx --config_path jinaai/jina-embeddings-v2-base-code/ --data_path demo/demo.txt --dtype float16 --batch-size 2 --target sdaa
        ```

        推理结果：

        ```bash
        cos_sim: 0.7282363772392273
        ```

   模型推理参数说明：
   
   | 参数 | 说明 | 默认值 |
   | ------------- | ------------- | ------------- |
   | data_path  |数据路径 |N/A|
   | ckpt       | 模型onnx路径  | N/A |
   | batch-size | 推理的batch_size  | 1 |
   | target     | 推理的设备 | `sdaa` |
   | dtype      | 模型推理使用的数据类型  | float16 |
   | model_name | 模型名称 | `jina-embeddings-v2-base-code` |


### 2.6 精度验证

请提前准备数据集，执行以下命令，获得推理精度数据。

```bash
python example_valid.py --model_name jina-embeddings-v2-base-code --ckpt jina-embeddings-v2-base-code_dyn_fp16.onnx --config_path jinaai/jina-embeddings-v2-base-code/ --data_path code_search_net --dtype float16 --batch-size 32 --target sdaa
```

精度结果如下：

```shell
eval_metric: 0.9042207792207793
summary: avg_sps: 48.44136519410522, e2e_time: 457.7905662059784, data_time: 3.8057243824005127, avg_inference_time: 0.6180288891168827, avg_preprocess_time: 3.3763579991988805e-05, avg_postprocess: 3.8899728574106016e-05
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