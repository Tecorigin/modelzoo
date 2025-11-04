## 基础环境安装

使用本仓库运行模型推理，您需要完成相应的环境依赖检查和环境安装。整体流程如下：
1. 检查环境依赖：检查环境安装的相应依赖条件是否满足。
2. 构建仓库环境：拉取仓库并更新仓库中的子模块。
3. 构建Docker环境：使用Dockerfile创建模型推理时所需的Docker环境。

### 1. 环境依赖

请确保您的服务器上已经安装以下环境依赖：
- 系统架构：X86（64位）。
- 系统版本：Ubuntu22.04。
- 安装Docker应用程序。
- 服务器已安装太初TecoDriver。使用以下命令检查服务器是否已经安装TecoDriver。如果未安装，请参考太初[《环境安装手册》中的 TecoDriver安装指南](http://docs.tecorigin.net/release/software_installation/)，安装与仓库版本一致的TecoDriver。

  ```shell
  dpkg -l | grep tecodriver
  ```

### 2. 构建仓库环境

1. 执行以下命令，下载仓库。

   ```shell
   git clone https://gitee.com/tecorigin/modelzoo.git
   ```

### 2. 构建Docker环境

使用Dockerfile创建模型推理时所需的Docker环境。
1. 执行以下命令，进入Dockerfile所在目录。 
   ```shell
   cd <modelzoo_dir>/TecoInference
   ```

2. 执行以下命令，构建名为`model_infer`的镜像。
   ```shell
   DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build -t model_infer -f Dockerfile ../
   ```

3. 执行以下命令，启动容器。
   ```shell
   docker run -itd -v <path_to_host_workspace>:<path_to_docker_workspace> --name model_infer --net=host --add-host=<host_name>:<host_ip> --pid=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2    --device /dev/tcaicard3 --privileged=true --shm-size 128g model_infer
   ```

   容器配置参数说明如下：
   | 参数名 | 说明 |
   | ------------- | ------------- |
   | -v | 用于将宿主机上指定的目录映射到容器内的目录。映射的路径下需要包含权重文件和数据集（如果需要验证精度）。 |
   | --name | Docker容器名称。 |
   | --device | 映射的太初加速卡数量，根据配置酌情修改。若使用单卡4个计算设备进行推理，请按照上述命令样式进   行配置。 |
   | --privileged | 是否允许容器访问主机所有设备。 |
   |--shm-size | 设置share memory大小。
   | --add-host | 设置主机名和IP地址映射 | --add-host=host_name:host_ip, host_name和host_ip可以用`hostname`和`hostname -I`查看|


## 3. 常见问题

### 如何解决安装依赖时报错？

安装依赖时可能因网络问题失败，可多尝试几次或按照下述过程替换源：

   ```
    pip install xxx -i <source-link>
   ```

### 遇到环境报错，如何解决？

需要执行`source /opt/tecoai/setvars.sh`, 执行一次即可。您也可以将执行命令加到 ~/.bashrc中。

### 导入paddle找不到libssl.so.1.1的问题？

1. 在环境中找到libssl.so.1.1文件路径并添加至LD_LIBRARY_PATH中

   ```
   find ~ -name libssl.so.1.1
   export LD_LIBRARY_PATH=FILE_PATH:$LD_LIBRARY_PATH:
   ```
   其中，`FILE_PATH`替换为libssl.so.1.1的文件夹路径 `xxx/lib/`。

2. 若本地没有, 则执行以下命令，尝试安装：
   ```
   wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
   sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
   ```