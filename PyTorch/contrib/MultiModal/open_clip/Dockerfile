# 指定基础镜像
ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

# 设置工作目录
WORKDIR /workspace/

# 添加依赖文件（可选，如果你提前生成好了）
ADD requirements.txt /workspace/

# 修复 sh 链接并切换 shell 为 bash
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh

# 设置 shell 为 bash，后续命令以 bash 执行
SHELL ["/bin/bash", "-c"]

# 激活 conda 虚拟环境并安装依赖
RUN conda activate torch_env && \
    pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt || true

# 添加 open_clip 项目到容器中
ADD . /workspace/open_clip/

# 切换到训练脚本所在目录
WORKDIR /workspace/open_clip/run_scripts/

# 默认入口（可以修改train_data.csv路径）
CMD ["python", "run_demo.py", "--train-data=/workspace/coco/train_data.csv"]


