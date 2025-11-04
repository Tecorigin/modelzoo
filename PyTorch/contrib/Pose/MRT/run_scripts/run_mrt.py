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

if __name__ == '__main__':
    args = parse_args()
    
    model_name = args.model_name
    epoch = args.epoch
    bs = args.batch_size
    device = args.device
    eval = args.eval
    autocast = args.autocast
    checkpoint = args.checkpoint
    data_path = args.data_path
    ddp = args.ddp

    if model_name != 'mrt':
        raise ValueError('model_name should be mrt')
    
    # training
    if not eval:
        cmd = f'python train_mrt.py \
                --epoch {epoch} \
                --bs {bs} \
                --data_path {data_path} \
                --device {device}'
        if autocast:
            cmd += ' --autocast'
        if ddp:
            cmd += ' --ddp'
    # evaluation
    else:
        cmd = f'python test_mrt.py \
                --device {device} \
                --data_path {data_path} \
                --checkpoint {checkpoint}'
        if autocast:
            raise ValueError('autocast is not supported for evaluation')
        if not checkpoint:
            raise ValueError('checkpoint should be provided for evaluation')
    os.system(cmd)
