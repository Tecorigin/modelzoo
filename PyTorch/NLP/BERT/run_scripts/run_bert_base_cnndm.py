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
    lr = args.lr
    num_workers = args.num_workers
    device = args.device
    autocast = args.autocast
    grad_scale = args.grad_scale
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    early_stop = args.early_stop
    warmup = args.warm_up
    max_seq_length = args.max_seq_length
    ckpt_path = args.checkpoint_path
    do_predict= args.do_predict
    global_batch_size = bs * nproc_per_node * nnode


    project_path = str(Path(__file__).resolve().parents[1])

    hpyer_params=f'--bert_model {model_name} \
            --bert_model {model_name} \
            --output_dir ./log/cnndm \
            --init_checkpoint {ckpt_path} \
            --train_file={data_path}/cnndm \
            --predict_file={data_path}/cnndm \
            --max_seq_length {max_seq_length} \
            --do_train \
            --train_batch_size={bs} \
            --predict_batch_size={bs} \
            --learning_rate {lr} \
            --warmup_proportion {warmup} \
            --seed 666 \
            --max_steps {step} \
            --device {device} \
            --loss_scale 65536 \
            --config_file ./bert_configs/base.json'

    if nnode > 1:
        raise Exception("Recent task do not support nnode > 1. Set --nnode=1 !")

    if nnode == 1 and nproc_per_node>1:
        cmd = f'SDAA_RUNTIME_AUTOFALLBACK=1 SDAA_FALLBACK_OPS=index,_index_put_impl_ torchrun --nproc_per_node {nproc_per_node} --master-port 29505 {project_path}/pipeline/run_cnndm.py \
            {hpyer_params}'
        if autocast:
            cmd += ' --amp'
        if do_predict:
            cmd += ' --do_predict'
    else:
        cmd = f'SDAA_RUNTIME_AUTOFALLBACK=1 SDAA_FALLBACK_OPS=index,_index_put_impl_ python {project_path}/pipeline/run_cnndm.py \
            {hpyer_params}'
        if autocast:
            cmd += ' --amp'
        if do_predict:
            cmd += ' --do_predict'

    print_formatted_cmd(cmd)
    os.system(cmd)
