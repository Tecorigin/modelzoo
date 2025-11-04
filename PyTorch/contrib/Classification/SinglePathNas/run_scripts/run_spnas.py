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
import os

from argument import parse_args

if __name__ == "__main__":
    args = parse_args()

    os.chdir("..")
    cmd = (
        f"python train.py \
          --exp_name {args.exp_name} \
          --layers {args.layers} \
          --num_choices {args.num_choices} \
          --batch_size {args.batch_size} \
          --epochs {args.epochs} \
          --num_steps {args.num_steps} \
          --lr {args.lr} \
          --momentum {args.momentum} \
          --weight-decay {args.weight_decay} \
          --print_freq {args.print_freq} \
          --val_interval {args.val_interval} \
          --save_path {args.save_path} \
          --seed {args.seed} \
          --data_path {args.data_path} \
          --classes {args.classes} \
          --dataset {args.dataset} \
          --cutout \
          --cutout_length {args.cutout_length} \
          --auto_aug \
          --resize"
    )


    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)