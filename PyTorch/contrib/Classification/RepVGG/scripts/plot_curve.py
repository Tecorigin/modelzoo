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

def plot_loss_curve(log_file):
    """
    从日志文件中提取loss值并绘制曲线。

    Args:
        log_file: 日志文件路径。
    """

    iterations = []
    losses = []

    with open(log_file, 'r') as f:
        for line in f:
            # 使用正则表达式匹配包含loss信息的行
            match = re.search(r"Train: \[\d+/\d+\]\[(\d+)/(\d+)\]\t.*loss (\d+\.\d+) \((\d+\.\d+)\)", line)
            if match:
                current_iter = int(match.group(1))
                total_iter = int(match.group(2))
                loss = float(match.group(4))  # 使用平滑后的loss值

                iterations.append(current_iter)
                losses.append(loss)

    # 绘制loss曲线
    plt.figure(figsize=(10, 6))  # 调整图像大小以获得更好的可视化效果
    plt.plot(iterations, losses, marker='o', linestyle='-', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.grid(True)  # 添加网格线
    plt.xticks(range(0, max(iterations) + 1, 1000)) # 设置x轴刻度，每1000个iteration显示一个刻度

    # 保存图像
    plt.savefig('train_sdaa_3rd.png')
    print(f"Loss curve saved to train_sdaa_3rd.png")
    plt.show()

if __name__ == "__main__":
    log_file = 'train_sdaa_3rd.log'
    plot_loss_curve(log_file)