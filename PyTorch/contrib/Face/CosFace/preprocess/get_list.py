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
import argparse

def get_list_for_casia_webface112x96(data_folder, output_list_file):
    """
    功能：
    1. 找到 `data_folder`（CASIA-WebFace-112X96）的所有子文件夹（每个子文件夹代表一个 ID）。
    2. 排除与 LFW 重复的 ID（此处示例中为 ['0166921', '1056413', '1193098']）。
    3. 为每个子文件夹（ID）列出其中所有 .jpg 图片的绝对路径，并标注标签（从 0 开始）。
    4. 将这些信息按行写入 `output_list_file`，格式为： `绝对路径 label`。
    """

    # 1) 获取所有子文件夹名称（排除非文件夹）
    all_subfolders = []
    for d in os.listdir(data_folder):
        d_full = os.path.join(data_folder, d)
        if os.path.isdir(d_full):
            all_subfolders.append(d)

    # 2) 排除可能与 LFW 重复的 ID
    exclude_ids = {'0166921', '1056413', '1193098'}
    subfolders = [sf for sf in all_subfolders if sf not in exclude_ids]

    # 3) 打开输出文件，准备写入
    with open(output_list_file, 'w') as f:
        total_folders = len(subfolders)
        # 4) 遍历每个子文件夹
        for i, sub in enumerate(subfolders):
            print(f"Collecting the {i+1}th folder (total {total_folders}) ...")

            subfolder_path = os.path.join(data_folder, sub)
            # 5) 获取该子文件夹下所有 .jpg 文件
            file_list = [fn for fn in os.listdir(subfolder_path)
                         if fn.lower().endswith('.jpg')]

            # 6) 写入到列表文件：每行 = 图片绝对路径 + 空格 + 标签
            for fn in file_list:
                file_path = os.path.join(subfolder_path, fn)
                label = i  # Python中索引从0开始，即可与 MATLAB 中 i-1 对应
                f.write(f"{file_path} {label}\n")

def main():
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Generate the file list for CASIA-WebFace-112X96 dataset."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="/data/datasets/CosFace/result/CASIA-WebFace-112X96",
        help="CASIA-WebFace-112X96 文件夹的路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/data/datasets/CosFace/CASIA-WebFace-112X96.txt",
        help="输出文件的路径"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用生成列表的函数
    get_list_for_casia_webface112x96(args.data_folder, args.output_file)

if __name__ == "__main__":
    main()