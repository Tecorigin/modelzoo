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
import os
import matplotlib.pyplot as plt

def parse_log_file(log_file_path):
    iters = []
    losses = []
    
    # 正则表达式匹配iter数和loss值
    pattern = r'Epoch: \[\d+\]\s+\[\s*(\d+)/\d+\].*?loss: [\d.]+ \(([\d.]+)\)'
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iter_num = int(match.group(1))
                loss = float(match.group(2))
                iters.append(iter_num)
                losses.append(loss)
    
    return iters, losses

def plot_loss_curve(iters, losses, output_file='train_sdaa_3rd.png'):
    """绘制loss曲线并保存为图片"""
    plt.figure(figsize=(12, 6))
    plt.plot(iters, losses, 'b-', linewidth=1)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 如果数据点很多，可以适当调整x轴的显示密度
    if len(iters) > 100:
        step = len(iters) // 10
        plt.xticks(iters[::step])
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    output_path = os.path.abspath(output_path)
    plt.savefig(output_path)
    print(f"Loss curve saved to {output_file}")

if __name__ == "__main__":
    log_file = os.path.join(os.path.dirname(__file__), "train_sdaa_3rd.log")
    log_file = os.path.abspath(log_file)
    try:
        iters, losses = parse_log_file(log_file)
        if not iters:
            print("No valid data found in the log file.")
        else:
            print(f"Found {len(iters)} data points.")
            plot_loss_curve(iters, losses)
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")