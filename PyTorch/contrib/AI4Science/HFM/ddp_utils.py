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
from distutils.spawn import spawn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import torch
import torch_sdaa.core.sdaa_model as sm

def setup(world_size=1, rank=0,args=None):
    multi_machine = args.multi_machine
    node_rank = args.node_rank
    local_size = args.local_size
    MASTER_ADDR = args.master_addr
    MASTER_PORT = args.master_port
    device = sm.sdaa_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1' if multi_machine ==False else MASTER_ADDR
    os.environ['MASTER_PORT'] = '22412' if multi_machine ==False else MASTER_PORT
    rank = rank + node_rank*local_size if multi_machine else rank
    # initialize the process group
    if multi_machine==False:
        dist.init_process_group("tccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(
            "tccl", init_method="tcp://{}:{}".format(MASTER_ADDR, MASTER_PORT), rank=rank,
            world_size=world_size)

def cleanup():
    dist.destroy_process_group()



if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--local_world_size", type=int, default=1)
    # args = parser.parse_args()
    # spawn(args.local_world_size, args.local_rank)
