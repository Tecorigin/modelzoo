# 指定基础镜像（已集成 PyTorch + SDAA）
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# 设置工作目录
WORKDIR /workspace/

# 修复 sh 链接并切换 shell 为 bash
RUN rm -f /bin/sh && ln -s /bin/bash /bin/sh
SHELL ["/bin/bash", "-c"]

# 添加 Python 依赖文件
ADD requirements.txt /workspace/

# 激活 conda 虚拟环境 sd15 并安装依赖
RUN conda activate sd15 && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt || true

# 添加 Stable Diffusion 项目代码（注意路径根据你的项目结构调整）
ADD . /workspace/stable-diffusion-v1-5/

# 设置工作目录为训练脚本目录
WORKDIR /workspace/stable-diffusion-v1-5/run_scripts/

# 默认执行训练命令（可通过 --train-data 修改路径）
CMD ["python", "run_demo.py"]
