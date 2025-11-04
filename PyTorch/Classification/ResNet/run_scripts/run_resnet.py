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
from formate_cmd import print_formatted_cmd

if __name__ == '__main__':
    args = parse_args()
    args = check_argument(args)
    
    model_name = args.model_name
    epoch = args.epoch
    step = args.step
    bs = args.batch_size
    data_path = args.dataset_path
    nproc_per_node = args.nproc_per_node
    nnode = args.nnode
    precision_align = args.precision_align
    precision_align_cuda_path = args.precision_align_cuda_path
    precision_align_log_path = args.precision_align_log_path
    lr = args.lr
    num_workers = args.num_workers
    device = args.device
    profiler = args.profiler
    profiler_path = args.profiler_path
    autocast = args.autocast
    grad_scale = args.grad_scale
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    early_stop = args.early_stop
    mlperf = args.mlperf_mode
    warmup = args.warm_up
    optimizer = args.optimizer
    dataset = args.dataset
    compile = args.compile

    global_batch_size = bs * nproc_per_node * nnode

    if warmup <= -1:
        if global_batch_size<256:
            warmup = 0
        else:
            warmup = 2*(global_batch_size // 256 )
    
    
    project_path = str(Path(__file__).resolve().parents[1])

    if step >0:
        training_only=True
        no_ckpt=False
    else:
        training_only=False
        no_ckpt=True
    
    if device != 'sdaa':
        fused_optimizer = False
    else:
        fused_optimizer = True
    
    if precision_align_log_path is not None:
        os.environ['PT_PRECISION_TOOL_LOG'] = precision_align_log_path
    
    hyper_parameters = f'\
         --model {model_name} \
         --precision FP32 \
         --seed 1234 \
         --optimizer {optimizer} \
         --mode convergence \
         --platform sdaa \
         --data {data_path} \
         --epochs {epoch} \
         --mixup 0.0 \
         --label-smooth 0.1 \
         --batch-size {bs} \
         --lr {lr} \
         --warmup {warmup} \
         --workspace ./logs \
         --raport-file raport.json \
         --device {device} \
         --print-freq 1 \
         --drop_last \
         --grad_scaler {grad_scale} \
         --static-loss-scale 65536 \
         --channel_last \
         --training-only {training_only} \
         --no-checkpoints {no_ckpt} \
         --persistent_workers \
         --PrefetchedWrapper \
         --collate_fn \
         --amp {autocast} \
         --compile {compile} \
         --fused_optimizer {fused_optimizer} \
	     --prof {step} \
         --early_stop {early_stop} \
         --mlperf_mode {mlperf} \
         --pin_memory \
         --workers {num_workers} \
         --profiler {profiler} \
         --profiler_path {profiler_path} \
         --layer_diff {precision_align} \
         --dataset {dataset} \
    '
    if args.resume:
        hyper_parameters += f'--resume {args.resume} '
        
    env = 'SET_CONV2D_WEIGHT_CHWN=1 '
    
    if nnode == 1 and nproc_per_node>1:
        cmd = f'{env}  torchrun --master_port {master_port} --nproc_per_node {nproc_per_node} {project_path}/launch.py \
            {hyper_parameters} \
            '

    elif nnode>1:
        cmd = f'{env}   torchrun --nnode {nnode} --node_rank {node_rank} \
            --master_addr {master_addr} --master_port {master_port} --nproc_per_node {nproc_per_node} {project_path}/launch.py \
            {hyper_parameters} \
        '

    else:
        cmd = f'{env}   python   {project_path}/launch.py \
            {hyper_parameters} \
        '
    print_formatted_cmd(cmd)
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)

    
    
    
    