## 基础环境安装

使用本仓库运行模型训练或模型推理，您需要完成相应的环境依赖检查和环境安装。整体流程如下：

1. 检查环境依赖：检查环境安装的相应依赖条件是否满足。
2. 构建仓库环境：拉取仓库。

### 1.环境依赖

请确保您的服务器上已经安装以下环境依赖：

- 系统架构：X86 (64位) 
- 系统版本：Ubuntu22.04。
- 安装Docker应用程序，参考[文档](https://docs.docker.com/engine/install/)。
- 服务器已安装太初TecoDriver。使用以下命令检查服务器是否已经安装TecoDriver。如果未安装，请参考太初[《环境安装手册》中的TecoDriver安装指南](http://docs.tecorigin.net/release/software_installation/v1.1.0/#1674fea26c9011eebbbf0242ac110008)，安装与仓库版本一致的TecoDriver。

    ```
    dpkg -l | grep tecodriver
    ```

### 2.构建仓库环境

执行以下命令，下载仓库。

```
git clone https://gitee.com/tecorigin/modelzoo.git
```
