# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from server_models import CLIENTS

import os
import time
import pytest
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 根据实际路径进行调整
ONNX_PATH = os.environ.get('ONNX_PATH', '/mnt/test_models/onnx_models/NLP/')
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/mnt/checkpoint/TecoInferenceEngine/')

BGE_MODELS = [
    ['bge-large-zh-v1.5', 512, os.path.join(ONNX_PATH, 'BGE/bge-large-zh-v1.5/bge-large-zh-v1.5_dyn_fp16.onnx'), os.path.join(CONFIG_PATH, 'BGE/configs/bge-large-zh-v1.5'), '["样例数据-1", "样例数据-2"]'],
    ['bge-large-en-v1.5', 512, os.path.join(ONNX_PATH, 'BGE/bge-large-en-v1.5/bge-large-en-v1.5_dyn_fp16.onnx'), os.path.join(CONFIG_PATH, 'BGE/configs/bge-large-en-v1.5'), '["sample-data-1", "sample-data-2"]'],
    ['bge-m3', 512, os.path.join(ONNX_PATH, 'BGE/bge-m3/bge-m3_dyn_fp16.onnx'), os.path.join(CONFIG_PATH, 'BGE/configs/bge-m3'), '["样例数据-1", "样例数据-2"]'],
    ['bge-reranker-v2-m3', 512, os.path.join(ONNX_PATH, 'BGE/bge-reranker-v2-m3/bge-reranker-v2-m3_dyn_fp16.onnx'), os.path.join(CONFIG_PATH, 'BGE/configs/bge-reranker-v2-m3'), "[['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]"],
    ['bge-reranker-large', 512, os.path.join(ONNX_PATH, 'BGE/rerank/bge-reranker-large_float_dyn.onnx'), os.path.join(CONFIG_PATH, 'BGE/configs/bge-reranker-large'), "[['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]"],
    ['jina-embeddings-v2-base-code', 1024, os.path.join(ONNX_PATH, 'jina/jina-embeddings-v2-base-code_dyn_fp16.onnx'), os.path.join(CONFIG_PATH, 'jina-embeddings-v2-base-code/configs/'), "['How do I access the index while iterating over a sequence with a for loop?','# Use the built-in enumerator\\nfor idx, x in enumerate(xs):\\n    print(idx, x)',]"],
]

@pytest.mark.parametrize("model_name, max_length, model_path, config_path, sentences_list", BGE_MODELS)
def test_bge(model_name, max_length, model_path, config_path, sentences_list):

    try:
        server_cmd = f"python server.py --model_name {model_name} \
                        --model_file {model_path} "
        proc = subprocess.Popen(server_cmd, shell=True, close_fds=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        print('\n')
        start_server_time = time.time()
        while proc.poll() is None and time.time() - start_server_time < 600:
            output = proc.stdout.readline().decode()
            print(output, end='')
            if output and 'Create Session:' in output:
                time.sleep(0.1)
                break

        client = CLIENTS[model_name](config_path,
                                        max_length,
                                        model_name,
                                        )
        client(eval(sentences_list))

        proc.terminate()
    finally:
        if proc.poll() is None:
            proc.kill()