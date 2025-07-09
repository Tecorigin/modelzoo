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
import os
import argparse
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine
from utils.datasets.jina import create_dataloader

import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


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
    parser.add_argument('--batch-size', type=int, default=32)
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

    # load dataset
    dataloader = create_dataloader(args.data_path, args.config_path, args.batch_size, max_length=args.max_sequence_len)

    e2e_start_time = time.time()

    time_info = {
        "data_load":[],
        "pre_time":[],
        "post_time":[],
    }

    if args.target == "sdaa":
        pipeline.module.start_timing()
    else:
        time_info["infer_time"] = []

    query_embeddings = []
    code_embeddings = []
    start_data_time = time.time()
    for queries, code_snippets in tqdm(dataloader):
        time_info["data_load"].append(time.time() - start_data_time)
        start_time = time.time()

        queries_inputs = [v for k, v in queries.items()]
        code_inputs = [v for k, v in code_snippets.items()]

        # padding
        batch_nums = queries_inputs[0].shape[0]
        if batch_nums < args.batch_size:
            queries_inputs = [np.concatenate([a] + [a[-1:]]  * (args.batch_size - batch_nums)) for a in queries_inputs]
            code_inputs = [np.concatenate([a] + [a[-1:]] * (args.batch_size - batch_nums)) for a in code_inputs]
        time_info["pre_time"].append(time.time() - start_time)

        _, sentence_embedding_queries = pipeline(queries_inputs)
        _, sentence_embedding_code_snippets = pipeline(code_inputs)
        start_time = time.time()

        query_embeddings.append(sentence_embedding_queries[:batch_nums])
        code_embeddings.append(sentence_embedding_code_snippets[:batch_nums])

        time_info["post_time"].append(time.time() - start_time)
        if args.target != "sdaa":
            time_info["infer_time"].append(pipeline.run_time)
        start_data_time = time.time()

    if args.target == "sdaa":
        time_info["infer_time"] = [max(pipeline.module.get_infer_time()) / 1e3]
    
    time_info["e2e_time"] = time.time() - e2e_start_time

    query_embeddings = np.concatenate(query_embeddings, axis=0)
    code_embeddings = np.concatenate(code_embeddings, axis=0)

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(query_embeddings, code_embeddings)

    # 评估 Recall@1
    recall_at_1 = 0
    for i in range(len(similarity_matrix)):
        most_similar_code_index = np.argmax(similarity_matrix[i])
        if most_similar_code_index == i:
            recall_at_1 += 1
    recall_at_1 /= len(similarity_matrix)

    batchs = len(dataloader)
    samples = batchs * args.batch_size

    print("eval_metric:", recall_at_1)
    print(f'summary: avg_sps: {samples / time_info["e2e_time"]}, e2e_time: {time_info["e2e_time"]}, data_time: {sum(time_info["data_load"])}, avg_inference_time: {sum(time_info["infer_time"]) / batchs}, avg_preprocess_time: {sum(time_info["pre_time"]) / batchs}, avg_postprocess: {sum(time_info["post_time"]) / batchs}')

    # 释放device显存，stream等资源
    if "sdaa" in args.target:
        pipeline.release()
