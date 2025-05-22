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

if __name__ == "__main__":
    args = parse_args()
    args = check_argument(args)
    
    isRFN = args.RFN
    image_path_autoencoder = args.image_path_autoencoder
    image_path_rfn = args.image_path_rfn
    gray = args.gray
    model_name = args.model_name
    resume_nestfuse = args.resume_nestfuse
    resume_rfn = args.resume_rfn
    train_num = args.train_num
    device = args.device
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    
    torchrun_parmeters = f"\
        --image_path_autoencoder {image_path_autoencoder} \
        --gray {gray} \
        --train_num {train_num} \
        --device {device} \
        --batch_size {batch_size} \
        --num_epochs {num_epochs} \
        --lr {lr} \
        "

    env = ""

    project_path = str(Path(__file__).resolve().parents[1])

    cmd = f"{env} python {project_path}/run_train.py \
          {torchrun_parmeters} "
    
    print(cmd)
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)