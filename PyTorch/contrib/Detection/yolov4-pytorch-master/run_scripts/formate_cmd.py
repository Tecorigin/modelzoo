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


import re


def print_formatted_cmd(cmd):
    """
    解析并打印命令的环境变量和运行参数。

    参数：
    cmd (str)：要打印的命令。

    示例：
    >>> print_cmd("ENV=1 python script.py --arg1 value1 --arg2 value2")
    环境变量:
    ENV=1

    运行参数:
    arg1=value1
    arg2=value2
    """

    # 以 "python" 进行划分，获取环境变量和运行参数
    print('#' * 50 + ' RUN INFO ' + '#' * 50)
    if 'python3' in cmd:
        envs = cmd.split("python3")[0].strip()
        args = cmd.split("python3")[1].strip()
    elif 'python' in cmd:
        envs = cmd.split("python")[0].strip()
        args = cmd.split("python")[1].strip()
    elif 'torchrun' in cmd:
        envs = cmd.split("torchrun")[0].strip()
        args = cmd.split("torchrun")[1].strip()

    # 打印环境变量
    if envs != "":  # 去除无环境变量脚本
        envs = re.split(" ", envs)
        print("环境变量:")
        for _ in envs:
            if _ != "":  # 去除空值打印
                print(_)
        print("\n")

    # 打印运行参数
    if args != "":
        args = re.split("--", args)
        print("运行参数:")
        for _ in args[1:]:
            if _ != "":  # 去除空值打印
                _ = _.strip().replace(" ", "=")  # 去除首尾空格，并将空格替换为等号
                print(_)
        print("\n")
    print('#' * 50 + ' RUN INFO ' + '#' * 50)