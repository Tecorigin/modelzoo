# FastAPI for Tecoinference

## 简介
FastAPI 是一个现代、快速（高性能）的 Web 框架，用于构建 API。FastAPI 旨在简化开发过程，同时提供高性能和自动化的文档功能。本文档提供部分模型基于FastAPI的服务化示例。

## 环境准备

1. 请参考[基础环境安装](../../doc/Environment.md)，完成使用前的基础环境安装和检查。
2. 安装FastAPI依赖。

    ```bash
    conda activate tvm-build_py310
    pip install -r requirements.txt
    ```

## 模型支持列表

|模型名称|权重与配置文件参考|
|---|---|
| bge-large-zh-v1.5 | [README](../example/nlp/bge/docs/README_bge-large-zh-v1.5.md) |
| bge-large-en-v1.5 | [README](../example/nlp/bge/docs/README_bge-large-en-v1.5.md) |
| bge-m3 | [README](../example/nlp/bge/docs/README_bge-m3.md) |
| bge-reranker-v2-m3 | [README](../example/nlp/bge/docs/README_bge-reranker-v2-m3.md) |
| jina-embeddings-v2-base-code | [README](../example/nlp/jina/README.md) |
| bge-reranker-large | [README](../example/nlp/bge/docs/README_bge-reranker-large.md) |

## 启动FastAPI推理服务

请参考[模型支持列表](#模型支持列表)对应的README文档提前准备ONNX权重和模型配置文件。

1. 启动FastAPI服务

    执行以下命令启动FastAPI推理服务, 以`bge-large-zh-v1.5`为例:

    ```bash
    cd /modelzoo/TecoInference/fastapi
    python server.py --model_name bge-large-zh-v1.5 --model_file bge-large-zh-v1.5_dyn_fp16.onnx --ip localhost --save_engine True
    ```

    参数说明
    | 参数 | 说明 |
    | ------------- | ------------- |
    | model_name    | 模型名称, 参考模型支持列表 |
    | model_file    | 权重路径, 可以是onnx也可以是engine文件 |
    | ip            | 当前机器ip  |
    | port          | 端口号  |
    | save_engine   | 将生成的engine保存到本地 |

2. 推理测试

    新建终端执行以下命令, 测试推理:
    ```bash
    cd /modelzoo/TecoInference/fastapi
    python client.py --model_name bge-large-zh-v1.5 --config_path BAAI/bge-large-zh-v1.5 --ip localhost --sentences_list '["样例数据-1", "样例数据-2"]'
    ```

    终端有如下输出:
    ```bash
    2025-05-06 03:40:43 [INFO] ['样例数据-1', '样例数据-2'] cos_sim:0.8790898523166538
    ```

    参数说明
    | 参数 | 说明 |
    | ------------- | ------------- |
    | model_name    | 模型名称, 参考模型支持列表 |
    | config_path   | 模型配置文件路径 |
    | ip            | 启动服务的机器IP |
    | port          | 启动服务的端口号  |
    | sentences_list   | 测试的句子列表, 可以参考模型支持列表对应README中的demo示例 |
    | max_length   | 最大序列长度 |