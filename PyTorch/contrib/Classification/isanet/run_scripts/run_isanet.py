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

def build_hyper_parameters(args):
    # 提取并存储参数
    launcher = args.launcher
    amp = args.amp
    config = args.config
    cfg_options = args.cfg_options

    hyper_parameters = f"{config} --launcher {launcher} "

    if args.amp:
        hyper_parameters += f" --amp"
    if args.cfg_options:
        cfg_str = ""
        for k, v in args.cfg_options.items():
            # 确保值被正确格式化
            if isinstance(v, str) and ' ' in v:
                # 如果值包含空格，用引号包裹
                cfg_str += f"{k}='{v}' "
            else:
                cfg_str += f"{k}={v} "       
        hyper_parameters += f" --cfg-options {cfg_str.strip()} "

    return hyper_parameters

def build_command(args,hyper_parameters):

    nnodesx = args.nnodes
    node_rank = args.node_rank
    master_addr = args.master_addr
    master_port = args.master_port
    nproc_per_node = args.nproc_per_node
    cmd = ""
    cmd = f'python -m torch.distributed.launch --nnodes={nnodesx} \
        --node_rank={node_rank} \
        --master_addr={master_addr} \
        --nproc_per_node={nproc_per_node} \
        --master_port={master_port} \
        ../tools/train.py {hyper_parameters}'
    print("cmd--->>>>>:\n{}\n".format(cmd))
    return cmd

def excute_command(cmd):
    import subprocess
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        exit_code = e.returncode
        print("Command failed with exit code:", exit_code)
        exit(exit_code)

if __name__ == "__main__":
    args = parse_args()
    hyper_parameters = build_hyper_parameters(args)
    cmd = build_command(args,hyper_parameters)
    excute_command(cmd)

