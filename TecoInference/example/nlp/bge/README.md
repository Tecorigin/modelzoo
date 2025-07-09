# Bge

Bge系列模型均使用此文档

## 1. 环境准备

### 1.1 基础环境安装

请参考推理首页的[基础环境安装](../../../README.md)章节，完成推理前的基础环境检查和安装。

### 1.2 安装第三方依赖

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
   cd <modelzoo_dir>/TecoInference/example/nlp/bge
   ```

4. 执行以下命令，安装第三方依赖。

   ```shell
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。

## 2. 支持的模型
| MODELS  | Tecoinference|
| ------------- | ------------- |
| [bge-large-zh](./docs/README_bge-large-zh.md)|YES|
| [bge-reranker-large](./docs/README_bge-reranker-large.md)|YES|
| [bge-large-en-v1.5](./docs/README_bge-large-en-v1.5.md)|YES|
| [bge-large-zh-v1.5](./docs/README_bge-large-zh-v1.5.md)|YES|
| [bge-m3](./docs/README_bge-m3.md)|YES|
| [bge-reranker-v2-m3](./docs/README_bge-reranker-v2-m3.md)|YES|

可以点击对应链接查看具体使用教程。