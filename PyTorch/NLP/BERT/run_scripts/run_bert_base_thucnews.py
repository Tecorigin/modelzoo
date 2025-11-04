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
    max_seq_length = args.max_seq_length
    ckpt_path = args.checkpoint_path

    global_batch_size = bs * nproc_per_node * nnode


    project_path = str(Path(__file__).resolve().parents[1])

    if precision_align_log_path is not None:
        os.environ['PT_PRECISION_TOOL_LOG'] = precision_align_log_path

    hpyper_params=f'--bert_model {model_name} \
            --output_dir ./log/tnews \
            --init_checkpoint {ckpt_path} \
            --data_dir {data_path} \
            --max_seq_length {max_seq_length} \
            --task_name sst-2 \
            --do_train \
            --do_eval \
            --train_batch_size={bs} \
            --learning_rate {lr} \
            --max_steps {step} \
            --num_train_epochs {epoch} \
            --warmup_proportion {warmup} \
            --seed 666 \
            --device {device} \
            --loss_scale 65536 \
            --vocab_file ./vocab/bert_base_chinese_vocab.txt \
            --do_lower_case \
            --gradient_accumulation_steps 1\
            --config_file ./bert_configs/thucnews.json'

    # check
    if precision_align:
        raise Exception("Recent task do not support precision align. Set --precision_align=False !")

    if profiler:
        raise Exception("Recent task do not support profiler. Set --profiler=False !")

    if nnode > 1:
        raise Exception("Recent task do not support nnode > 1. Set --nnode=1 !")

    if nnode == 1 and nproc_per_node>1:
        cmd = f'SDAA_BERT_HIGHPERF=1 torchrun --nproc_per_node {nproc_per_node} {project_path}/pipeline/run_thucnews.py \
            {hpyper_params}'
        if autocast:
            cmd += ' --amp'
    else:
        cmd = f'SDAA_BERT_HIGHPERF=1 python {project_path}/pipeline/run_thucnews.py \
            {hpyper_params}'
        if autocast:
            cmd += ' --amp'

    print_formatted_cmd(cmd)
    os.system(cmd)
