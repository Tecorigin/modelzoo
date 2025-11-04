# === 指定基础镜像 ===
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# === 设置工作目录 ===
WORKDIR /workspace/

# === 设置 shell 为 bash（推荐）===
SHELL ["/bin/bash", "-c"]

# === 拷贝 requirements 文件（不要用 ADD，避免自动解压行为）===
COPY requirements.txt .

# === 拷贝 open_clip 项目代码 ===
COPY . /workspace/open_clip/

# === 安装依赖 ===
# ⚠️ 注意：必须使用 source activate，否则 conda 无效
RUN source activate torch_env_py310 && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# === 设置训练脚本路径为工作目录 ===
WORKDIR /workspace/open_clip/run_scripts/

# === 默认入口，指向 CSV 数据路径 ===
CMD ["python", "run_open_clip.py", "--train-data=/workspace/coco/train_data.csv"]
