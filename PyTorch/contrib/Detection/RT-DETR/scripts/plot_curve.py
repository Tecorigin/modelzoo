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
# OF SUCH DAMAGE


import re
import matplotlib.pyplot as plt


def plot_loss_curve(log_path, output_path):
    steps = []  # 用于存储步骤数（即训练步数）
    losses = []  # 用于存储对应的 loss 值

    # 正则表达式匹配日志行，例如：
    # Epoch: [0]  [   0/7393]  eta: 7:02:51  lr: 0.000010  loss: 42.3491 (42.3491)  loss_bbox: ...
    pattern = re.compile(
        r'Epoch: \[\d+\]\s+\[\s*(\d+)/\d+\].*?loss: [\d.]+ \(([\d.]+)\)'
    )

    # 读取日志文件，逐行匹配提取数据
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)

    # 绘制曲线
    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, linewidth=1.5)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 自动调整 x 轴刻度密度，最多显示 20 个刻度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f'Saved loss curve to {output_path}')


if __name__ == '__main__':
    plot_loss_curve(
        log_path='train_sdaa_3rd.log',
        output_path='train_sdaa_3rd.png'
    )
