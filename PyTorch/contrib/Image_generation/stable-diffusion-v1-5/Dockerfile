# === 基础镜像（已集成 PyTorch + SDAA）===
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# === 设置工作目录 ===
WORKDIR /workspace/

# === 切换 shell 为 bash（推荐写法）===
SHELL ["/bin/bash", "-c"]

# === 添加 requirements 文件（放最前避免无效层缓存）===
COPY requirements.txt .

# === 拷贝整个项目（可替换为 .dockerignore 控制）===
COPY . /workspace/stable-diffusion-v1-5/

# === 激活 conda 并安装依赖 ===
RUN source activate sd15 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# === 设置运行目录 ===
WORKDIR /workspace/stable-diffusion-v1-5/run_scripts/

# === 启动默认训练命令 ===
CMD ["python", "run_demo.py"]
