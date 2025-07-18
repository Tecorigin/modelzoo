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

import sys
import subprocess
from argument import parse_args

def main():
    # 解析传给 run_demo.py 的参数（实际是传给 main.py 的训练参数）
    args = parse_args(sys.argv[1:])

    # 构造训练命令，这里示例调用python main.py 并传入所有参数
    cmd = ["python", "../src/open_clip_train/main.py"]


    # 将 argparse Namespace 转成命令行参数列表
    for k, v in vars(args).items():
        if v is None:
            continue
        # 布尔值特殊处理，True 作为开关参数，False跳过
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k.replace('_', '-')}")
            continue

        # list 类型，转换为多次传参（如 force-image-size 可能是list）
        if isinstance(v, list):
            for item in v:
                cmd.append(f"--{k.replace('_', '-')}")
                cmd.append(str(item))
            continue

        # dict 类型（如 aug-cfg），转成 key=value 形式
        if isinstance(v, dict):
            for key, val in v.items():
                cmd.append(f"--{k.replace('_', '-')}")
                cmd.append(f"{key}={val}")
            continue

        # 普通参数，直接传入
        cmd.append(f"--{k.replace('_', '-')}")
        cmd.append(str(v))

    # 打印命令方便调试
    print("执行命令：", " ".join(cmd))

    # 使用 subprocess 执行命令，shell=False更安全
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"训练脚本执行失败，错误码：{e.returncode}")
        exit(e.returncode)

if __name__ == "__main__":
    main()
