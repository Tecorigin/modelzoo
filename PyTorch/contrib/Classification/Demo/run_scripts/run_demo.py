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

from argument import parse_args ,check_argument
import os
from pathlib import Path

if __name__ == "__main__":
    args = parse_args()
    args = check_argument(args)
    
    model_name = args.model_name
    epoch = args.epoch
    bs = args.batch_size
    device = args.device

    nnode = args.nnode
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    nproc_per_node = args.nproc_per_node

    global_batch_size = bs * nproc_per_node * nnode

    torchrun_parmeters = f"\
        --nnodes {nnode} \
        --node_rank {node_rank} \
        --master_addr {master_addr} \
        --master_port {master_port} \
        --nproc_per_node {nproc_per_node}\
        "
    hyper_parameters = f"\
         --total_epochs {epoch} \
         --batch_size {bs} \
         --device {device} \
         --save_every 1 \
    "

    env = ""

    project_path = str(Path(__file__).resolve().parents[1])

    cmd = f"{env}  torchrun {torchrun_parmeters} {project_path}/multigpu_torchrun.py \
        {hyper_parameters} \
            "
    print(cmd)
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)
    
    
    
    