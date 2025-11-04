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

import matplotlib.pyplot as plt
import re

def plot_loss_curve(log_file, output_file):
    """
    从日志文件中提取 Loss 值并绘制曲线。

    Args:
        log_file: 日志文件路径。
        output_file: 输出图像文件路径。
    """

    iterations = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 使用正则表达式匹配包含迭代次数和 Loss 值的行
            match = re.search(r"Train: \d+ \[ *(\d+)/10009.*?Loss: ([\d.]+) \(", line)
            if match:
                iteration = int(match.group(1))
                loss = float(match.group(2))
                iterations.append(iteration)
                losses.append(loss)
                
    # 确保有数据被提取
    if not iterations or not losses:
        print(f"Error: No data found in {log_file} matching the expected format.")
        return

    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(iterations, losses, marker='o', linestyle='-', markersize=3) #marker='o'显示数据点，linestyle='-'显示连接线，markersize=3设置点的大小
    plt.xlabel('Iteration (xxx/10009)')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.grid(True) #显示网格
    plt.tight_layout() #自动调整子图参数，使之填充整个图像区域
    plt.savefig(output_file)
    plt.show() #显示图像

if __name__ == "__main__":
    log_file1 = "train_sdaa_3rd_MobileOne.log"  # 你的日志文件
    output_file1 = "train_sdaa_3rd_MobileOne.png"  # 输出图像文件名
    plot_loss_curve(log_file1, output_file1)
    print(f"Loss curve saved to {output_file1}")

    log_file2 = "train_sdaa_3rd_MobilenetV1.log"  # 你的日志文件
    output_file2 = "train_sdaa_3rd_MobilenetV1.png"  # 输出图像文件名
    plot_loss_curve(log_file2, output_file2)
    print(f"Loss curve saved to {output_file2}")

    log_file3 = "train_sdaa_3rd_MobilenetV2.log"  # 你的日志文件
    output_file3 = "train_sdaa_3rd_MobilenetV2.png"  # 输出图像文件名
    plot_loss_curve(log_file3, output_file3)
    print(f"Loss curve saved to {output_file3}")