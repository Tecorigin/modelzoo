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

import os
import argparse
from server_models import CLIENTS

# 测试调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bge-large-zh-v1.5', choices=CLIENTS.keys())
    parser.add_argument('--config_path', type=str, default='BAAI/bge-large-zh-v1.5')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--sentences_list', type=str, default='["样例数据-1", "样例数据-2"]', help="list of sentences or sentences file path")
    parser.add_argument('--max_length', type=int, default=512, help="max sequence length, must same as server shape")

    args = parser.parse_args()

    client = CLIENTS[args.model_name](args.config_path,
                                      args.max_length,
                                      args.model_name,
                                      ip=args.ip,
                                      port=args.port
                                      )

    if os.path.exists(args.sentences_list):
        with open(args.sentences_list, "r") as f:
            sentences_lists = f.readlines()
        for sentences_list in sentences_lists:
            client(eval(sentences_list))
    else:
        client(eval(args.sentences_list))