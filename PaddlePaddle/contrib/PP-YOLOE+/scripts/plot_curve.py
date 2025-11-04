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

def parse_log_file(log_file_path):
    """
    解析日志文件，提取迭代数和损失值。
    """
    iter_nums = []
    losses = {"loss": [], "loss_cls": [], "loss_iou": [], "loss_dfl": []}
    
    with open(log_file_path, "r") as f:
        for line in f:
            # 使用正则表达式匹配日志中的关键信息
            # match = re.search(r"\[.*([0-9]+)/[0-9]+\] .* loss: ([0-9\.]+) loss_cls: ([0-9\.]+) loss_iou: ([0-9\.]+) loss_dfl: ([0-9\.]+)", line)
            match = re.search(r"Epoch: \[([0-9]+)\] \[\s*([0-9]+)\s*/([0-9]+)\s*\].*loss: ([0-9\.]+).*loss_cls: ([0-9\.]+).*loss_iou: ([0-9\.]+).*loss_dfl: ([0-9\.]+)", line)
            if match:
                epoch = int(match.group(1))
                iter_num = int(match.group(2))  # 提取迭代数
                total_iter = int(match.group(3))
                loss = float(match.group(4))
                loss_cls = float(match.group(5))
                loss_iou = float(match.group(6))
                loss_dfl = float(match.group(7))
                
                iter_nums.append(epoch*total_iter+iter_num)
                losses["loss"].append(loss)
                losses["loss_cls"].append(loss_cls)
                losses["loss_iou"].append(loss_iou)
                losses["loss_dfl"].append(loss_dfl)
    
    return iter_nums, losses

def plot_loss_curves(iter_nums, losses, output_file):
    """
    绘制损失曲线并保存为 PNG 文件。
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制每种损失的曲线
    for loss_name, loss_values in losses.items():
        plt.plot(iter_nums, loss_values, label=loss_name)
    
    # 添加图例和标签
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.grid(True)
    
    # 保存为 PNG 文件
    plt.savefig(output_file)
    print(f"Loss curves saved to {output_file}")



if __name__ == "__main__":
    log_file_path = "train_sdaa_3rd.log"  # 替换为你的日志文件路径
    output_file = "train_sdaa_3rd.png"  # 输出文件名
    
    iter_nums, losses = parse_log_file(log_file_path)
    plot_loss_curves(iter_nums, losses, output_file)



