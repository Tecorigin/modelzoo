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
import torch

from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH
from utils.preprocess.pytorch.ssd import preprocess
from utils.postprocess.pytorch.ssd import postprocess


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
    

def get_detection_results(prec, conf_thres=0.2):
    class_names = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                   'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
                   'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                   'sofa', 'train', 'tvmonitor']
    result_str = ""
    box_num = prec['boxes'].shape[0]
    for i in range(box_num): 
        confidence = prec['scores'][i]
        if confidence < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, prec['boxes'][i])
        class_id = prec['labels'][i]
        class_name = class_names[int(class_id)]  # 根据类别编号映射名称
        result_str += f"目标 {i + 1}:"
        result_str += f" - 类别: {class_name}"
        result_str += f" - 坐标: ({x1}, {y1}, {x2}, {y2})"
        result_str += f" - 置信度: {confidence:.2f}\n"
    return result_str


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='./ssd_dyn_float16.onnx', help='onnx path')
    parser.add_argument('--input_name', type=str, default='input', help='input name')
    parser.add_argument('--data_path', type=str, default='./images', help='images path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--input_size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--target', default='onnx', help='sdaa or onnx')
    parser.add_argument('--conf_thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--dtype', type=str, default='float16', help='use FP16 half-precision inference')
    parser.add_argument('--pass_path', type=str, default=PASS_PATH / 'default_pass.py', help='pass_path for tvm')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    RANK = int(os.environ.get('OMPI_COMM_WOLRD_RANK',0))
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

    for idx, file_name in enumerate(os.listdir(opt.data_path)):
        file_path = os.path.join(opt.data_path, file_name)
        images = preprocess(file_path, half=opt.dtype=="float16", resize_shape=opt.input_size)
        prec = infer_engine(images)
        prec = (torch.from_numpy(prec[:, :, :21]).float(), torch.from_numpy(prec[:, :, 21:]).float())
        prec = postprocess(prec)
        results = get_detection_results(prec[0], opt.conf_thres)
        print(f"第{int(idx)+1}张图片：{file_name}")
        print(results)

    