# 指定基础镜像
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# 设置工作目录
WORKDIR /workspace/

# 添加 requirements.txt
ADD requirements.txt /workspace/

# 修复 sh 链接并切换 shell 为 bash
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh

# 设置默认 Python 环境路径
ENV CONDA_ENV_NAME=torch_env_py310
ENV CONDA_ENV_PATH=/opt/conda/envs/${CONDA_ENV_NAME}
ENV PATH=${CONDA_ENV_PATH}/bin:$PATH

# 安装 Python 依赖（使用指定环境的 pip）
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 添加 ControlNet 项目
ADD . /workspace/Controlnet/

# 设置项目运行目录
WORKDIR /workspace/Controlnet/run_scripts/

# 设置默认启动命令（可修改参数）
CMD ["python", "run_controlnet.py"]
