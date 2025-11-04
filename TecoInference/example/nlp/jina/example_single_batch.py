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

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine

import torch
import numpy as np
from transformers import AutoTokenizer
from numpy.linalg import norm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='jina-embeddings-v2-base-code')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default="configs")
    parser.add_argument('--card_id', type=int, default=0, help='card id for engine inference')
    parser.add_argument('--target', type=str, default='sdaa', choices=['sdaa', 'onnx'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max_sequence_len', type=int, default=1024)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--save_engine', type=str2bool, default=False, help='save engine file')
    parser.add_argument('--pass_path', type=str, default=None)
    args = parser.parse_args()

    MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))
    work_bs = max(args.batch_size // MAX_ENGINE_NUMS, 1)
    input_size = [[work_bs, args.max_sequence_len], [work_bs, args.max_sequence_len]]

    pipeline = TecoInferEngine(
            model_name=args.model_name,
            ckpt=args.ckpt,
            batch_size=args.batch_size,
            input_size=input_size,
            target=args.target,
            dtype=args.dtype,
            pass_path=args.pass_path,
            save_engine=args.save_engine,
        )
    if args.ckpt.endswith(".engine"):
        pipeline.init_module(args.card_id)

    cos_sim = lambda a,b: (a @ b.permute(*torch.arange(b.ndim - 1, -1, -1))) / (norm(a)*norm(b))

    tokenizer = AutoTokenizer.from_pretrained(args.config_path)

    with open(args.data_path) as f:
        sentences = f.read()
    sentences = tokenizer(eval(sentences), truncation=True, padding="max_length", max_length=args.max_sequence_len)

    sentences_input = [np.array(v) for k, v in sentences.items()]
    _, sentence_embedding = pipeline(sentences_input)

    sentence_embedding = torch.from_numpy(sentence_embedding).float()

    print("cos_sim:", cos_sim(sentence_embedding[0], sentence_embedding[1]).item())

    # 释放device显存，stream等资源
    if "sdaa" in args.target:
        pipeline.release()
