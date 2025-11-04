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
import sys
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine

from utils.datasets.bge import create_dataloader
from utils.preprocess.pytorch.bge import preprocess
from utils.postprocess.pytorch.bge import postprocess

MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def infer_large_zh(pipeline, demo_inputs, opt):
    for sentences, inputs in demo_inputs.items():
        inputs = preprocess(inputs, opt.batch_size, batch_padding=True)
        sentence_embeddings = pipeline(inputs)
        sentence_embeddings = [sentence_embeddings[0], sentence_embeddings[1]]
        print(f"{sentences}-cosine_sim:{postprocess(sentence_embeddings, opt.batch_size, opt.model_name, demo=True)[0]}")

def infer_embedding(pipeline, demo_inputs, opt):
    for sentences, inputs in demo_inputs.items():
        inputs = preprocess(inputs, opt.batch_size, batch_padding=True)
        sentence_embeddings = pipeline(inputs[:2])
        sentence_embeddings = [sentence_embeddings[1][0], sentence_embeddings[1][1]]
        print(f"{sentences}-cosine_sim:{postprocess(sentence_embeddings, opt.batch_size, opt.model_name, demo=True)[0]}")

def infer_reranker_large(pipeline, demo_inputs, opt):
    for sentences, inputs in demo_inputs.items():
        inputs = preprocess(inputs, opt.batch_size, batch_padding=True)
        outputs = pipeline(inputs)
        print(f"{sentences}-相似度:{postprocess(outputs, opt.batch_size, opt.model_name)[0]}")

def inference(opt):

    # 加载数据集
    demo_inputs = create_dataloader(opt)

    # 初始化模型
    if opt.model_name == "bge-large-zh":
        input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 512]] * 3
    else:
        input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 512]] * 2
    pipeline = TecoInferEngine(ckpt=opt.ckpt,
                                model_name="bge",
                                target=opt.target,
                                batch_size=opt.batch_size,
                                input_size=input_size,
                                dtype="float16" if opt.half else "float32",
                                pass_path=opt.pass_path)
    if opt.ckpt.endswith(".engine"):
        pipeline.init_module(opt.card_id)

    if opt.model_name == "bge-large-zh":
        infer_large_zh(pipeline, demo_inputs, opt)
    elif opt.model_name in ["bge-reranker-large", 'bge-reranker-v2-m3']:
        infer_reranker_large(pipeline, demo_inputs, opt)
    else:
        infer_embedding(pipeline, demo_inputs, opt)

    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        pipeline.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='onnx path')
    parser.add_argument('--config-path', type=str, default=None, help='tokenizer config path')
    parser.add_argument('--data-path', type=str, default=None, help='dataset path')
    parser.add_argument('--model_name', type=str, default=None, help='model name for of yoloair, support "bge-large-zh", "bge-reranker-large"')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--shape', type=int, default=512, help='inference sequence_length')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--target', default='sdaa', help='sdaa、cpu、cuda or onnx')
    parser.add_argument('--card_id', type=int, default=0, help='card id for engine inference')
    parser.add_argument('--pass_path', type=str, default=None, help='pass_path for tvm')
    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()

    inference(opt)
