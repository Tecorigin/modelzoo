ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/
ADD requirements.txt /workspace/
RUN rm -rf /bin/sh && ln -s /bin/bash /bin/sh
RUN source activate torch_env && pip install --no-cache-dir -r requirements.txt
ADD . /workspace/Classification/deit/
WORKDIR /workspace/Classification/deit/

ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/pytorch:2.1.1-torch_sdaa2.1.1
FROM ${FROM_IMAGE_NAME}