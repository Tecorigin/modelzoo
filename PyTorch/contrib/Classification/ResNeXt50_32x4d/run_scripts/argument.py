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
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # 我认为有用的参数 (一部分是原来就有的, 另一部分是从 timm 上拿过来的)
    parser.add_argument("--epochs", type=int, default=300, metavar="N", help="number of epochs to train (default: 300)")
    parser.add_argument("--batch_size", type=int, default=128, help="Input batch size for training (default: 128)")
    parser.add_argument(
        "--precision", default="float32", type=str, help="Numeric precision. One of (float32, float16, bfloat16)"
    )
    parser.add_argument("--device", default="sdaa", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--max_step", default=-1, type=int)

    # 路径问题
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument(
        "--pretrained_path",
        default=None,
        type=str,
        help="Load this checkpoint as if they were the pretrained weights (with adaptation).",
    )
    parser.add_argument("--save_path", type=str, default="", help="Path to save checkpoints.")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)")
    parser.add_argument("--lrf", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Optimizer momentum (default: 0.9)")
    parser.add_argument(
        "--amp", action="store_true", default=False, help="use NVIDIA Apex AMP or Native AMP for mixed precision training"
    )

    opt = parser.parse_args()
    return opt
