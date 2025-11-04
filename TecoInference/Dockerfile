ARG FROM_IMAGE_NAME=jfrog.tecorigin.net/tecotp-docker/release/ubuntu22.04/x86_64/inference:1.2.0-teco_inference1.2.0

FROM ${FROM_IMAGE_NAME}
WORKDIR /workspace/
ADD . /workspace/modelzoo/

# tecoexec
RUN source activate tvm-build && cd /workspace/modelzoo/TecoInference/tecoexec && pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN sed -i '$a export LD_LIBRARY_PATH=/root/anaconda3/envs/tvm-build/lib/python3.8/site-packages/tvm/:$LD_LIBRARY_PATH' ~/.bashrc
RUN cd /workspace/modelzoo/TecoInference/tecoexec/ && wget http://jfrog.tecorigin.net/artifactory/tecobs-ap-teco_inference/artifacts/1.2.0/release/1.2.0/149/tecoexec && chmod +x tecoexec
RUN sed -i '$a export TECOEXEC_PATH=/workspace/modelzoo/TecoInference/tecoexec/' ~/.bashrc
WORKDIR /workspace/modelzoo/TecoInference/

