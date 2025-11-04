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
import matplotlib.pyplot as plt

def plot_loss_curve(log_file, output_file):
    """
    从日志文件中提取 loss 数据并绘制曲线图。

    Args:
        log_file (str): 日志文件路径。
        output_file (str): 输出图像文件路径。
    """

    iterations = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 使用正则表达式匹配包含 iteration 和 loss 的行
            match = re.search(r"Epoch: \[\d+\]\s+\[\s*(\d+)/(\d+)\][\s\S]*loss: (\d+\.\d+)\s+\((\d+\.\d+)\)", line)
            if match:
                iteration = int(match.group(1))
                total_iterations = int(match.group(2))
                loss = float(match.group(4))  # 使用 (loss) 中的值
                iterations.append(iteration)
                losses.append(loss)

    if not iterations:
        print(f"没有在 {log_file} 中找到匹配的 loss 数据。请检查日志文件格式。")
        return

    plt.figure(figsize=(10, 6))  # 设置图像大小
    plt.plot(iterations, losses, marker='o', linestyle='-') # 添加 marker 使点更清晰
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)  # 添加网格
    plt.xticks(range(0, max(iterations) + 1, max(iterations)//10)) # 设置x轴刻度，避免刻度过于密集
    plt.savefig(output_file)
    plt.show() #显示图像

if __name__ == "__main__":
    log_file = "train_sdaa_3rd.log"  # 你的日志文件
    output_file = "train_sdaa_3rd.png"  # 输出图像文件
    plot_loss_curve(log_file, output_file)
    print(f"Loss 曲线图已保存到 {output_file}")