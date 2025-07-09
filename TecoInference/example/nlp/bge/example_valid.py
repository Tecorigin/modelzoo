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
import time
import copy
import argparse
from tqdm import tqdm
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
from mldr import eval_bge_reranker_v2_m3
from tools import eval_bge_large_zh_v15, eval_bge_large_en_v15, eval_bge_m3

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

def diff2(eval_data, base_data):
    # 均方相对误差开方
    return ((np.abs(eval_data - base_data)**2).sum() / (base_data**2).sum())**0.5

def get_error(opt, preds):

    base = np.load(os.path.join(opt.onnx_datas, f"{opt.model_name}/{opt.model_name}_bs{opt.batch_size}_onnx.npy"))
    
    preds = np.asanyarray(preds).astype("float32")
    base = np.asanyarray(base[:len(preds)]).astype("float32")
    print(f"eval_metric:{diff2(preds, base)}")

def infer_large_zh(pipeline, validation_loader, opt):
    preds = []
    e2e_time, pre_time, run_time, post_time, ips = [], [], [], [], []
    for batch_i, batch in enumerate(tqdm(validation_loader)):
        sentence_embeddings = []
        for inputs in batch:
            start_time = time.time()
            batch_size_data = inputs['input_ids'].shape[0]
            inputs = preprocess(inputs, opt.batch_size, batch_padding=True)
            preprocess_time = time.time() - start_time
            # TODO share memory bug
            tmp_output = pipeline(inputs)
            output_embeddings = copy.deepcopy(tmp_output)
            sentence_embeddings.append(output_embeddings)
            # sentence_embeddings.append(pipeline(inputs))
            infer_time = time.time() - start_time

            postprocess_time = infer_time - pipeline.run_time - preprocess_time
            sps = opt.batch_size / infer_time
            e2e_time.append(infer_time)
            pre_time.append(preprocess_time)
            run_time.append(pipeline.run_time)
            post_time.append(postprocess_time)
            ips.append(sps)

        preds.extend(postprocess(sentence_embeddings, batch_size_data, opt.model_name))

    if opt.target == "sdaa":
        get_error(opt, np.array(preds))
    else:
        np.save(f"./datas/{opt.model_name}_bs{opt.batch_size}_{opt.target}.npy", np.array(preds))
    
    count = len(ips)
    print(f'summary: avg_sps: {sum(ips)/count} images/s, e2e_time: {sum(e2e_time)} s, avg_inference_time: {sum(run_time[5:])/(count-5)} s, avg_preprocess_time: {sum(pre_time)/count} s, avg_postprocess: {sum(post_time)/count} s')

def infer_reranker_large(pipeline, validation_loader, opt):
    preds = []
    e2e_time, pre_time, run_time, post_time, ips = [], [], [], [], []

    for batch_i, batch in enumerate(tqdm(validation_loader)):
        start_time = time.time()
        batch_size_data = batch['input_ids'].shape[0]
        inputs = preprocess(batch, opt.batch_size, batch_padding=True)
        preprocess_time = time.time() - start_time

        outputs = pipeline(inputs)
        infer_time = time.time() - start_time

        preds.extend(postprocess(outputs, batch_size_data, opt.model_name))

        postprocess_time = infer_time - pipeline.run_time - preprocess_time
        sps = opt.batch_size / infer_time
        e2e_time.append(infer_time)
        pre_time.append(preprocess_time)
        run_time.append(pipeline.run_time)
        post_time.append(postprocess_time)
        ips.append(sps)

    if opt.target == "sdaa":
        get_error(opt, np.array(preds))
    else:
        np.save(f"./datas/{opt.model_name}_bs{opt.batch_size}_{opt.target}.npy", np.array(preds))
    
    count = len(ips)
    print(f'summary: avg_sps: {sum(ips)/count} images/s, e2e_time: {sum(e2e_time)} s, avg_inference_time: {sum(run_time[5:])/(count-5)} s, avg_preprocess_time: {sum(pre_time)/count} s, avg_postprocess: {sum(post_time)/count} s')

def inference(opt):

    # 初始化模型
    if opt.model_name == "bge-large-zh":
        input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 512]] * 3
    else:
        input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 512]] * 2
    pipeline = TecoInferEngine(ckpt=opt.ckpt,
                                model_name="bge" if opt.model_name != 'bge-reranker-v2-m3' else opt.model_name,
                                target=opt.target,
                                batch_size=opt.batch_size,
                                input_size=input_size,
                                dtype="float16" if opt.half else "float32",
                                pass_path=opt.pass_path,
                                save_engine=opt.save_engine)
    if opt.ckpt.endswith(".engine"):
        pipeline.init_module(opt.card_id)
    
    if opt.model_name == "bge-large-zh":
        validation_loader = create_dataloader(opt)
        infer_large_zh(pipeline, validation_loader, opt)
    elif opt.model_name == "bge-reranker-large":
        validation_loader = create_dataloader(opt)
        infer_reranker_large(pipeline, validation_loader, opt)
    elif opt.model_name == "bge-large-zh-v1.5":
        eval_bge_large_zh_v15(pipeline, opt)
    elif opt.model_name == "bge-large-en-v1.5":
        eval_bge_large_en_v15(pipeline, opt)
    elif opt.model_name == "bge-m3":
        eval_bge_m3(pipeline, opt)
    elif opt.model_name == "bge-reranker-v2-m3":
        eval_bge_reranker_v2_m3(pipeline, opt)
    
    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        pipeline.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None, help='onnx path')
    parser.add_argument('--config-path', type=str, default=None, help='tokenizer config path')
    parser.add_argument('--onnx-datas', type=str, default=None, help='onnx-eval-datas')
    parser.add_argument('--data-path', type=str, default=None, help='dataset path')
    parser.add_argument('--file-type', type=str, default='json', help='dataset file type')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--shape', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--target', default='sdaa', help='sdaa、cpu、cuda or onnx')
    parser.add_argument('--pass_path', type=str, default=None, help='pass_path for tvm')
    parser.add_argument('--save_engine', type=str2bool, default=False, help='save engine file when use trt')
    parser.add_argument('--card_id', type=int, default=0, help='card id for engine inference')
    parser.add_argument('--model_name', type=str, default=None, 
            choices=["bge-large-zh", "bge-reranker-large", "bge-large-zh-v1.5", "bge-large-en-v1.5", "bge-m3", "bge-reranker-v2-m3"], help='model name of bge')
    opt = parser.parse_args()

    return opt

if __name__ == "__main__":
    opt = parse_opt()

    inference(opt)
