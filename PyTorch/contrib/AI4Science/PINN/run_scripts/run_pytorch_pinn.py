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
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="wave", help="模型名称")
    parser.add_argument("--epoch", type=int, default=2048, help="训练轮次，和训练步数冲突")
    parser.add_argument("--batch_size", type=int, default=512, help="每一轮的批次大小")
    parser.add_argument("--device", type=str, default="sdaa", help="指定运行设备")
    parser.add_argument("--show_graph", type=str, default="False", help="查看偏微分方程解的图像")
    parser.add_argument('--nproc_per_node', required=False, default=1, type=int,
                        help="在执行分布式数据并行时每个 node 上的 rank 数量, 不输入时默认为1, 表示单核执行")
    parser.add_argument('--nnode', required=False, default=1, type=int,
                        help="用于执行分布式数据并行训练时 node 的数量")
    args = parser.parse_args()

    model_name = args.model_name
    epoch = args.epoch
    batch_size = args.batch_size
    device = args.device
    show_graph = args.show_graph
    nproc_per_node = args.nproc_per_node
    nnode = args.nnode
              
    os.system("torchrun" + " --nnode=" + str(nnode) + " --nproc_per_node=" + str(nproc_per_node) + 
              " ../model/" + "solve_" + model_name + "_equation.py" + 
              " --epoch=" + str(epoch) + " --batch_size=" + str(batch_size) + 
              " --device=" + device + " --show_graph=" + show_graph)

if __name__ == "__main__":
    main()
