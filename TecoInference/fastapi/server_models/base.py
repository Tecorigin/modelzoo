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

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # modelzoo root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT / "TecoInference"))  # add ROOT to PATH
from engine.tecoinfer_pytorch import TecoInferEngine

import logging
import uvicorn
import requests

from fastapi import FastAPI

app = FastAPI()
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

class BaseServer():
    def __init__(self,
                 model_name,
                 onnx_path,
                 input_shape,
                 batch_size=None,
                 pass_path=None,
                 dtype='float16',
                 save_engine=False,
                 card_id=0
                 ):

        self.logger = logging.getLogger(f"{model_name}_server")
        if batch_size is None:
            batch_size = input_shape[0][0]

        self.logger.info(f"Init {model_name} model")
        self.model = TecoInferEngine(ckpt=onnx_path,
                                model_name=model_name,
                                target='sdaa',
                                batch_size=batch_size,
                                input_size=input_shape,
                                dtype=dtype,
                                pass_path=pass_path,
                                save_engine=save_engine)
        if onnx_path.endswith(".engine"):
            self.model.init_module(card_id)
        
        self.logger.info(f"Init {model_name} model Done")
        
    def start_server(self, ip='localhost', port=8000):
        uvicorn.run(app, host=ip, port=port)


class BaseClient():
    def __init__(self,
                 model_name,
                 ip="localhost",
                 port=8000,
                 ):
        self.model_name = model_name
        self.logger = logging.getLogger(f"{self.model_name}_client")
        self.logger.info("Clienter start init")
        self.url = f"http://{ip}:{port}/{model_name}"

    def client(self, data):
        # 发送 POST 请求
        response = requests.post(self.url, json=data)
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            response.raise_for_status()
