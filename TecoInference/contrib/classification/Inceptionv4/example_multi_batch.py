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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH
from utils.preprocess.pytorch.classification import preprocess
from utils.postprocess.pytorch.classification import postprocess


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./inceptionv4_float16_dyn.onnx', help='onnx path')
    parser.add_argument('--data_path', type=str, default='./images/', help='images path')
    parser.add_argument('--input_name', type=str, default='inceptionv4', help='input name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--input_size', type=int, default=224, help='inference size (pixels)')
    parser.add_argument('--target', default='sdaa', help='sdaa or cpu')
    parser.add_argument('--dtype', type=str, default='float32', help='use FP16 half-precision inference')
    parser.add_argument('--divide', type=bool, default=True, help='using mpi_run when set False')
    parser.add_argument('--pass_path', type=str, default=PASS_PATH / 'resnet/resnet_pass.py', help='pass_path for tvm')
    parser.add_argument('--topk', type=int, default=1, help='topk class')
    parser.add_argument('--verbose', action="store_true", help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    ###  init infer engine
    MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))
    input_size = [[max(opt.batch_size // MAX_ENGINE_NUMS, 1), 3, opt.input_size, opt.input_size]]
    infer_engine = TecoInferEngine(ckpt=opt.ckpt,
                                  batch_size=opt.batch_size,
                                  input_size=input_size,
                                  input_name=opt.input_name,
                                  target=opt.target,
                                  dtype=opt.dtype,
                                  backend='tvm',
                                  pass_path=opt.pass_path,
                                  )
    idx = 0
    for file_name in os.listdir(opt.data_path):
        file_path = os.path.join(opt.data_path, file_name)
        try:
            images = preprocess(file_path, dtype=opt.dtype)
            prec = infer_engine(images)
            output = postprocess(prec, target=opt.target, topk=opt.topk)
            print('第{}张 {} :'.format(idx, file_name), output)
            idx += 1
        except IOError:
            print("无法打开图片文件:", file_path)

    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        infer_engine.release()
