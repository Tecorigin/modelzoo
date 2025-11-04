# 动态基础镜像
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# 设置工作目录
WORKDIR /workspace/

# 切换到 bash 并安装依赖
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh
RUN source activate torch_env_py310

# 添加项目文件
ADD . /workspace/DeHazing/Dehamer/

# 切换到项目目录
WORKDIR /workspace/DeHazing/Dehamer/

# 暴露端口
EXPOSE 22
EXPOSE 8080
EXPOSE 8888

# 挂载点（可选，运行时仍需通过 docker run 指定）
VOLUME ["/mnt", "/mnt_qne00"]

# 默认启动命令
CMD ["/bin/bash"]
