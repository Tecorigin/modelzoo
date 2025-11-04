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

from argument import parse_args, check_argument
import os
from pathlib import Path

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    Epoch = args.epoch
    batch_size = args.batch_size
    nproc_per_node = args.nproc_per_node
    use_amp = args.use_amp
    use_ddp = args.use_ddp
    if use_ddp:
        local_rank = args.local_rank
    else:
        local_rank = 0
    classes_path = args.classes_path

    project_path = str(Path(__file__).resolve().parents[1])

    common_params = f"--model_name={model_name} \
        --epoch={Epoch} \
        --batch_size={batch_size} \
        --nproc_per_node={nproc_per_node} \
        --use_amp={use_amp}\
        --use_ddp={use_ddp}\
        --classes_path={classes_path}"

    if use_ddp:
        cmd = f"python -m torch.distributed.launch --nproc_per_node={nproc_per_node}  --master_port=25641  {project_path}/train.py {common_params}"
    else:
        cmd = f"python {project_path}/train.py {common_params}"

    print(cmd)
    os.system(cmd)


