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

from .base import app
from .bge import BgeServer, BgeClient, NumpyRequest


SERVERS = {
    'bge-large-zh-v1.5': BgeServer,
    'bge-large-en-v1.5': BgeServer,
    'bge-m3': BgeServer,
    'bge-reranker-v2-m3': BgeServer,
    'jina-embeddings-v2-base-code': BgeServer,
    'bge-reranker-large': BgeServer,
}

SHAPES = {
    'bge-large-zh-v1.5': [[2, 512], [2, 512]],
    'bge-large-en-v1.5': [[2, 512], [2, 512]],
    'bge-m3': [[2, 512], [2, 512]],
    'bge-reranker-v2-m3': [[1, 512], [1, 512]],
    'bge-reranker-large': [[1, 512], [1, 512]],
    'jina-embeddings-v2-base-code': [[2, 1024], [2, 1024]],
}

CLIENTS = {
    'bge-large-zh-v1.5': BgeClient,
    'bge-large-en-v1.5': BgeClient,
    'bge-m3': BgeClient,
    'bge-reranker-v2-m3': BgeClient,
    'bge-reranker-large': BgeClient,
    'jina-embeddings-v2-base-code': BgeClient,
}