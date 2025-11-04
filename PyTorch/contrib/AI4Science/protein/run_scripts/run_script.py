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

from argument import parse_args
import os
from pathlib import Path

if __name__ == '__main__':
    args = parse_args()
    
    device = args.device
    save_path = args.save_path
    k_fold = args.k_fold
    train_size = args.train_size
    save_rounds = args.save_rounds
    batch_size = args.batch_size
    epoch = args.epoch
    nproc_per_node = args.nproc_per_node

    project_path = str(Path(__file__).resolve().parents[1])


    if args.ddp:
        cmd = f'torchrun {project_path}/main.py \
                --ddp \
                --device {device} \
                --save_path {save_path} \
                --k_fold {k_fold} \
                --batch_size {batch_size} \
                --train_size {train_size} \
                --save_rounds {save_rounds} \
                --epoch {epoch} \
                --nproc_per_node {nproc_per_node} \
            '

    else:
        cmd = f'python  {project_path}/main.py \
                --device {device} \
                --save_path {save_path} \
                --k_fold {k_fold} \
                --batch_size {batch_size} \
                --train_size {train_size} \
                --save_rounds {save_rounds} \
                --epoch {epoch} \
            '
    os.system(cmd)