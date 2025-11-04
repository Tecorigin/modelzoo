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

from .base import BaseServer, BaseClient

import torch
import numpy as np
import torch.nn.functional as F

from pydantic import BaseModel
from transformers import AutoTokenizer
from numpy.linalg import norm

PASS_NAME = {
    'bge-large-zh-v1.5': 'bge',
    'bge-large-en-v1.5': 'bge',
    'bge-m3': 'bge',
    'bge-reranker-v2-m3': 'bge-reranker-v2-m3',
    'bge-reranker-large': 'bge',
    'jina-embeddings-v2-base-code': 'jina-embeddings-v2-base-code',
}


class NumpyRequest(BaseModel):
    input_ids_list: list
    attention_mask_list: list


class BgeServer(BaseServer):
    def __init__(self, model_name, *args, **kwargs):
        self.model_name = model_name
        super().__init__(PASS_NAME[model_name], *args, **kwargs)
        
        self.infer_fn_dict = {
            'bge-large-zh-v1.5': self.infer_bge_embedding,
            'bge-large-en-v1.5': self.infer_bge_embedding,
            'bge-m3': self.infer_bge_embedding,
            'bge-reranker-v2-m3': self.infer_bge_reranker,
            'bge-reranker-large': self.infer_bge_reranker,
            'jina-embeddings-v2-base-code': self.infer_jina_embedding,
        }

        self.infer_fn = self.infer_fn_dict[self.model_name]

    
    def get_input(self, request: NumpyRequest):
        """
        request to numpy data
        """
        input_ids = np.array(request.input_ids_list)
        attention_mask = np.array(request.attention_mask_list)

        self.logger.info(f"input_ids shape:{input_ids.shape}; attention_mask shape:{attention_mask.shape}")

        return [input_ids, attention_mask]

    def infer_bge_embedding(self, request: NumpyRequest):
        """
        inference bge-large-zh-v1.5, bge-large-en-v1.5 and bge-m3
        """
        model_inputs = self.get_input(request)
        model_output = self.model(model_inputs)

        sentence_embeddings = torch.from_numpy(model_output[0][:, 0]).float()
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        self.logger.info(f"sentence_embeddings shape:{sentence_embeddings.shape}")
        return {"sentence_embeddings": sentence_embeddings.tolist()}

    def infer_bge_reranker(self, request: NumpyRequest):
        """
        inference bge-reranker-v2-m3, bge-reranker-large
        """
        model_inputs = self.get_input(request)
        model_output = self.model(model_inputs)

        scores = torch.from_numpy(model_output)

        self.logger.info(f"scores shape:{scores.shape}")
        return {"scores": scores.tolist()}
    
    def infer_jina_embedding(self, request: NumpyRequest):
        """
        inference jina-embeddings-v2-base-code
        """
        model_inputs = self.get_input(request)
        model_output = self.model(model_inputs)

        token_embeddings = torch.from_numpy(model_output[0]).float()
        attention_mask = torch.from_numpy(model_inputs[1])

        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        self.logger.info(f"embeddings shape:{embeddings.shape}")
        return {"sentence_embeddings": embeddings.tolist()}

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        mean_pooling for jina-embeddings-v2-base-code
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class BgeClient(BaseClient):
    def __init__(self, config_path, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config_path)
        self.cos_sim = lambda a,b: (a @ b.permute(*torch.arange(b.ndim - 1, -1, -1))) / (norm(a)*norm(b))
        self.max_length = max_length
        self.logger.info("Clienter init done")

    def sentences2token(self, sentences):
        """
        sentences to tokens
        """
        encoded_input = self.tokenizer(sentences, padding='max_length', max_length=self.max_length, truncation=True)
        input_ids = np.array(encoded_input['input_ids'])
        attention_mask = np.array(encoded_input['attention_mask'])
        return input_ids, attention_mask

    def encode(self, sentences):
        input_ids, attention_mask = self.sentences2token(sentences)
        out = self.client({
            'input_ids_list':input_ids.tolist(),
            'attention_mask_list':attention_mask.tolist()
        })
        if 'reranker' in self.model_name:
            return out['scores']
        return torch.from_numpy(np.array(out['sentence_embeddings']))

    def __call__(self, sentences_list:list):
        """
        args:
            sentences_list: list of sentences, [["样例数据-1", "样例数据-2"],]  or ["样例数据-1", "样例数据-2"]
        """
        assert len(sentences_list) > 0, "len(sentences_list) need > 0 !"
        if not isinstance(sentences_list[0], list):
            sentences_list = [sentences_list]
        for sentences in sentences_list:
            if 'reranker' in self.model_name:
                scores = self.encode([sentences])
                self.logger.info(f"{sentences} scores:{scores}")
            else:
                sentence_embeddings = self.encode(sentences)
                self.logger.info(f"{sentences} cos_sim:{self.cos_sim(sentence_embeddings[0], sentence_embeddings[1]).item()}")