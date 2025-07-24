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
# argument.py
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="统一训练接口参数解析")

    # 模型路径 默认为 run_scripts 的上一级目录（即 stable-diffusion-v1-5）
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--sd_path", type=str, default=default_root, help="Stable Diffusion 模型目录")

    # COCO 数据集路径 必须用户自行指定，没有默认值
    parser.add_argument("--coco_img_root", type=str, required=True, help="COCO图片根目录")
    parser.add_argument("--coco_ann_path", type=str, required=True, help="COCO注释文件路径")

    parser.add_argument("--model_name", type=str, default="sd_unet", help="模型名称")
    parser.add_argument("--batch_size", type=int, default=1, help="训练批大小")
    parser.add_argument("--max_iter", type=int, default=100, help="最大训练迭代次数")
    parser.add_argument("--device", type=str, default="sdaa:0", help="训练设备，例 sdaa:0 或 cuda:0")
    parser.add_argument("--data_size", type=int, default=2000, help="训练用数据集大小")
    parser.add_argument("--log_path", type=str, default=None, help="日志文件路径，默认sd_path下的sdaa.log")
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径，默认sd_path下的unet_finetuned.pth")
    parser.add_argument("--accum_steps", type=int, default=2, help="梯度累积步数")

    args = parser.parse_args()

    # 设置默认日志和模型保存路径
    if args.log_path is None:
        args.log_path = os.path.join(args.sd_path, "sdaa.log")
    if args.save_path is None:
        args.save_path = os.path.join(args.sd_path, "unet_finetuned.pth")

    return args
