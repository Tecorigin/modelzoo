ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/paddle:1.0.0-paddle_sdaa1.0.0
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/
ADD requirements.txt /workspace/
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh
RUN source activate paddle_env && pip install --no-cache-dir -r requirements.txt
ADD . /workspace/Classification/ResNet/
WORKDIR /workspace/Classification/ResNet/
