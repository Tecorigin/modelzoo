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
    fp64 = args.precision_align


    project_path = str(Path(__file__).resolve().parents[1])

    if 'resnet' not in model_name:
        raise ValueError('please use resnet model')
    layers = model_name.split('t')[1]

    if autocast:
        yaml_path = f'{project_path}/ppcls/configs/ResNet{layers}_amp_O1.yaml'
    else:
        yaml_path = f'{project_path}/ppcls/configs/ResNet{layers}.yaml'
    if precision_align:
        yaml_path = f'{project_path}/ppcls/configs/ResNet{layers}.yaml'

    if step >0:
        eval_during_train = False
    else:
        eval_during_train = True

    if device=='sdaa':
        paddle_sdaa_env = 'PADDLE_XCCL_BACKEND=sdaa HIGH_PERFORMANCE_CONV=1 FLAGS_use_stream_safe_cuda_allocator=0 '
    else:
        paddle_sdaa_env = ''

    if bs==32 and nproc_per_node==4:
        env = 'FLAGS_set_to_1d=0 '
    else:
        env = 'FLAGS_set_to_1d=0   RANDOM_ALIGN_NV_DEVICE=a100 '

    global_batch_size = bs * nproc_per_node * nnode
    if global_batch_size<256:
        warmup = 0
    else:
        warmup = 2*(global_batch_size // 256 )

    if precision_align_log_path is not None:
        os.environ['PD_PRECISION_TOOL_LOG'] = precision_align_log_path

    hyper_parameters = f'\
            -o  DataLoader.Train.sampler.batch_size={bs} \
            -o  Global.device={device} \
            -o  Arch.data_format="NHWC" \
            -o  Global.print_batch_step=1 \
            -o  Global.epochs={epoch} \
            -o  Global.prof={step} \
            -o  Optimizer.lr.learning_rate={lr} \
            -o  Optimizer.lr.warmup_epoch={warmup} \
            -o  Global.eval_during_train={eval_during_train} \
            -o  DataLoader.Train.dataset.image_root={data_path} \
            -o  DataLoader.Train.dataset.cls_label_path={data_path}/train_list.txt \
            -o  DataLoader.Eval.dataset.image_root={data_path} \
            -o  DataLoader.Eval.dataset.cls_label_path={data_path}/val_list.txt \
            -o  Global.early_stop={early_stop} \
            -o  DataLoader.Train.loader.num_workers={num_workers} \
            -o  Global.profiler={profiler} \
            -o  Global.profiler_path={profiler_path} \
            -o  Global.precision_align={precision_align} \
            -o  Global.FP64={fp64} \
            -o  Global.precision_align_path={precision_align_cuda_path} \
            '


    if nnode == 1 and nproc_per_node>1:
        cmd = f'{paddle_sdaa_env} \
             {env} python -m paddle.distributed.launch {project_path}/tools/train.py \
            -c  {yaml_path} \
            {hyper_parameters} \
            '
    elif nnode>1:
        cmd = f'{paddle_sdaa_env} \
            {env} python -m paddle.distributed.launch --ips={master_addr} {project_path}/tools/train.py \
            -c  {yaml_path} \
            {hyper_parameters} \
            '
    else:
        cmd = f' {env} python  {project_path}/tools/train.py \
            -c   {yaml_path} \
            {hyper_parameters} \
            '
    print('本次运行命令',cmd)
    print_formatted_cmd(cmd)
    os.system(cmd)
