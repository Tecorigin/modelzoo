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
import time
from tqdm import tqdm
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
import torch
from torch.utils.data import DataLoader

from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH
from utils.preprocess.pytorch.ssd import preprocess
from utils.postprocess.pytorch.ssd import postprocess, get_metrics
from utils.datasets.ssd_dataset import VOCDataset, collate_fn


def str2bool(v):
    """
    将命令行输入的str转换为布尔值
    :param v: str值
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def main(opt):
    predictions = {}
    e2e_time = []
    pre_time = []
    run_time = []
    post_time = []
    ips = []

    RANK = int(os.environ.get('OMPI_COMM_WOLRD_RANK',0))
    max_step = int(os.environ.get("TECO_INFER_PIPELINES_MAX_STEPS", -1))
    warmup_step = int(os.environ.get("TECO_INFER_PIPELINES_WARMUP_STEPS", 0))
    global_step = 1
    iouv = torch.linspace(0.5, 0.95, 10, device='cpu')  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    ###  init infer engine
    MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

    input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 3, opt.input_size, opt.input_size]]
    infer_engine = TecoInferEngine(ckpt=opt.ckpt,
                                   batch_size=opt.batch_size,
                                   input_size=input_size,
                                   input_name=opt.input_name,
                                   target=opt.target,
                                   dtype=opt.dtype,
                                   pass_path=opt.pass_path,
                                   rank=RANK
                                   )

    # load dataset
    dataset = VOCDataset(opt.data_path, opt.input_size)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)

    seen = 0

    TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'  # tqdm bar format
    pbar = tqdm(dataloader, desc="inferencing", bar_format=TQDM_BAR_FORMAT)  # progress bar

    while True:
        # start infer
        for batch_i, batch in enumerate(pbar):
            start_time = time.time()
            images, targets, image_ids = batch
            images = preprocess(images, half=opt.dtype=="float16")
            preprocess_time = time.time() - start_time
            
            prec = infer_engine(images)
            model_time = infer_engine.run_time
            prec = (torch.from_numpy(prec[:, :, :21]).float(), torch.from_numpy(prec[:, :, 21:]).float())

            prec = postprocess(prec)
            infer_time = time.time() - start_time
            postprocess_time = infer_time - preprocess_time - model_time
            sps = opt.batch_size / infer_time

            predictions.update(
                {int(img_id): result for img_id, result in zip(image_ids, prec)}
            )

            if global_step > warmup_step:
                if opt.verbose:
                    print(f'e2e_time: {infer_time}, inference_time: {model_time}, preprocess_time: {preprocess_time}, postprocess: {postprocess_time}, sps: {sps}')
                e2e_time.append(infer_time)
                pre_time.append(preprocess_time)
                run_time.append(model_time)
                post_time.append(postprocess_time)
                ips.append(sps)

            if max_step > 0 and global_step == max_step:
                break
            global_step += 1
            
        if global_step >= max_step:
            break
    

    metrics = get_metrics(dataset, predictions)

    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        infer_engine.release()

    len = global_step - 1
    print("mAP@0.5~0.95: ", metrics['metrics']['mAP'])
    print(f'summary: avg_sps: {sum(ips)/len}, e2e_time: {sum(e2e_time)}, avg_inference_time: {sum(run_time[5:])/(len-5)}, avg_preprocess_time: {sum(pre_time)/len}, avg_postprocess: {sum(post_time)/len}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./ssd_dyn_float16.onnx', help='onnx path')
    parser.add_argument('--input_name', type=str, default='input', help='input name')
    parser.add_argument('--data_path', type=str, default='/mnt/nvme/common/train_dataset/voc/VOC2007', help='images path')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--input_size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
    parser.add_argument('--target', default='onnx', help='sdaa or cpu')
    parser.add_argument('--dtype', type=str, default='float16', help='use FP16 half-precision inference')
    parser.add_argument('--pass_path', type=str, default=PASS_PATH / 'default_pass.py', help='pass_path for tvm')
    parser.add_argument('--verbose', type=str2bool, default=False, help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    