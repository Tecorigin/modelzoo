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
from pathlib import Path
import os

if __name__ == '__main__':
    args = parse_args()
    autocast = args.autocast
    distributed = args.distributed
    work_dir = args.path

    project_path = str(Path(__file__).resolve().parents[1])

    if distributed:
        torchrun_parameter = f'python -m torch.distributed.launch' \
                             f' --nproc_per_node={args.nproc_per_node} ' \
                       
    else:
        torchrun_parameter = f'python ' \
    
    general_parameter = f' --dataset_path {args.dataset_path} ' \
                        f' --num_classes {args.num_classes} ' \
                        f' --model_name {args.model_name} ' \
                        f' --distributed {args.distributed} ' \
                        f' --autocast {args.autocast} ' \
                        f' --device {args.device} ' \
                        f' --epochs {args.epochs} ' \
                        f' --lr {args.lr} ' \
                        f' --batch_size {args.batch_size} '  \
                        f' --weights {args.weights} ' \
                        f' --step {args.step} ' \
        # Use DDP
    if distributed:
        # Use torchrun to start DDP
        cmd = torchrun_parameter + \
              f'{project_path}/train.py ' +\
              general_parameter
        
    else:
        # Training with single GPU
        cmd = f'python {project_path}/train.py ' +\
              general_parameter
    
    if work_dir:
        # Check working directory
        if os.path.exists(work_dir):
            print("====注意, 你选择的work_dir已经存在, 之前保存的checkpoints可能会被覆盖====")
        else:
            # create work_dir if not exist
            os.makedirs(work_dir)
            print(f"work_dir: [ {work_dir} ] is created successfully")
        cmd += f' --path {work_dir}'


    print(cmd)
    os.system(cmd)
