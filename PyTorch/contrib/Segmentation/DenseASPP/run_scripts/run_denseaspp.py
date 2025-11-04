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

from arguments import parse_args
from pathlib import Path
import os

if __name__ == '__main__':
    args = parse_args()
    use_amp = args.use_amp
    
    test_only = args.test_only
    distributed = args.distributed
    work_dir = args.work_dir
    nnodes = args.nnodes

    project_path = str(Path(__file__).resolve().parents[1])

    if distributed and nnodes > 1:
        torchrun_params = f'torchrun ' \
                             f' --master_port={args.master_port} ' \
                             f' --nproc_per_node={args.nproc_per_node} ' \
                             f' --nnodes={args.nnodes} ' \
                             f' --node_rank={args.nnodes} ' \
                             f' --master_addr={args.master_addr} ' 
                       
    else:
        torchrun_params = f'torchrun ' \
                             f' --master_port={args.master_port} ' \
                             f' --nproc_per_node={args.nproc_per_node} ' 
    
    common_hyper_params = f' --data_root {args.data_root} ' \
                            f' --dataset {args.dataset} ' \
                            f' --num_classes {args.num_classes} ' \
                            f' --model_name {args.model_name} ' \
                            f' --distributed {args.distributed} ' \
                            f' --use_amp {args.use_amp} ' \
                            f' --default_rank {args.default_rank} ' \
                            f' --crop_val {args.crop_val} ' \
                            f' --val_batch_size {args.val_batch_size} ' \
                            f' --crop_size {args.crop_size} ' \
                            f' --ckpt {args.ckpt} ' \
                            f' --loss_type {args.loss_type} ' \
                            f' --device {args.device} ' 
    
    train_hyper_params = f' --total_epochs {args.total_epochs} ' \
                            f' --optimizer {args.optimizer} ' \
                            f' --lr {args.lr} ' \
                            f' --lr_policy {args.lr_policy} ' \
                            f' --batch_size {args.batch_size} ' \
                            f' --continue_training {args.continue_training} ' \
                            f' --weight_decay {args.weight_decay} ' \
                            f' --random_seed {args.random_seed} ' \
                            f' --print_interval {args.print_interval} ' \
                            f' --val_epoch {args.val_epoch}' 
    

    # For training
    if not test_only:
        # Use DDP
        if distributed:
            # Use torchrun to start DDP
            cmd = torchrun_params + \
                  f'{project_path}/tools/train.py ' 
            
        else:
            # Training with single GPU
            cmd = f'python {project_path}/tools/train.py ' 
        
        cmd += train_hyper_params + \
               common_hyper_params
        
        if work_dir:
            # Check working directory
            if os.path.exists(work_dir):
                print("====注意, 你选择的work_dir已经存在, 之前保存的checkpoints可能会被覆盖====")
            else:
                # create work_dir if not exist
                os.makedirs(work_dir)
                print(f"work_dir: [ {work_dir} ] is created successfully")
            cmd += f' --work_dir {work_dir}'

    else:
        if distributed:
            # Use torchrun to start DDP
            cmd = torchrun_params + \
                  f'{project_path}/tools/test.py ' 
                  
        else:
            # Training with single GPU
            cmd = f'python {project_path}/tools/test.py ' 
        
        cmd += common_hyper_params

    print(cmd)
    os.system(cmd)
