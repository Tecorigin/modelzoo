# # BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# # reserved.
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# # Redistributions of source code must retain the above copyright notice,
# # this list of conditions and the following disclaimer.
# # Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# # Neither the name of the copyright holder nor the names of its contributors
# # may be used to endorse or promote products derived from this software
# # without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# # LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# # CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# # INTERRUPTION)
# # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# # STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# # WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# # OF SUCH DAMAGE.
# import argparse
# import os
# import os.path as osp
# import time
# import mmpretrain.visualization
# from copy import deepcopy
# from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法
# from mmengine.config import Config, ConfigDict, DictAction
# from mmengine.registry import RUNNERS, HOOKS
# from mmengine.runner import Runner
# from mmengine.utils import digit_version
# from mmengine.utils.dl_utils import TORCH_VERSION
# from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
# from mmengine.hooks import Hook
# from datetime import datetime

# BSD 3-Clause License
# Copyright (c) 2023, Tecorigin Co., Ltd. All rights reserved.
import loguru
from argument import parse_args
import subprocess
import os

def build_hyper_parameters(args):
    """构建训练超参数字符串"""
    # 只保留必要的参数，与test.sh保持一致
    hyper_parameters = f"{args.config}"  # 配置文件路径
    
    # 添加launcher参数
    hyper_parameters += f" --launcher {args.launcher}"
    
    # 添加amp参数
    if args.amp:
        hyper_parameters += " --amp"
    
    return hyper_parameters

def build_command(args, hyper_parameters):
    """构建完整的训练命令,与test.sh保持一致"""
      
    # 构建命令，使用 ../tools/train.py
    cmd = f'torchrun --master_port=29500 ../tools/train.py {hyper_parameters} 2>&1 | tee sdaa.log'
    
    # 格式化命令显示
    formatted_cmd = 'torchrun --master_port=29500 ../tools/train.py \\\n'
    parts = hyper_parameters.split(' --')
    formatted_cmd += '    --' + ' \\\n    --'.join(parts[1:])
    formatted_cmd += ' 2>&1 | tee sdaa.log'
    loguru.logger.info(f"将执行以下命令")
    loguru.logger.info(cmd)
    print("cmd--->>>>:")
    print(formatted_cmd)
    print()
    
    return cmd

def execute_command(cmd):
    """执行训练命令"""
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        # 等待进程完成
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code: {e.returncode}")
        exit(e.returncode)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit(1)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 构建训练参数
    hyper_parameters = build_hyper_parameters(args)
    
    # 构建完整命令
    cmd = build_command(args, hyper_parameters)
    
    # 执行命令
    execute_command(cmd)

if __name__ == '__main__':
    main()