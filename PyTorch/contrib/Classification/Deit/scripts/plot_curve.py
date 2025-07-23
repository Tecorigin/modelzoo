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

def parse_log_file(log_file):
    epochs = []
    iters = []
    losses = []
    global_iters = []
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    current_epoch = 0
    total_iters_per_epoch = 1251  # 根据您的日志，每epoch有1251个iter
    
    for line in lines:
        # 匹配epoch和iter信息
        epoch_match = re.search(r'Epoch: \[(\d+)\]', line)
        iter_match = re.search(r'\[(\s*\d+)/\d+\]', line)
        loss_match = re.search(r'loss: [\d.]+ \(([\d.]+)\)', line)
        
        if epoch_match and iter_match and loss_match:
            epoch = int(epoch_match.group(1))
            iter_num = int(iter_match.group(1).strip())
            loss = float(loss_match.group(1))
            
            # 计算全局iter数
            global_iter = epoch * total_iters_per_epoch + iter_num
            
            epochs.append(epoch)
            iters.append(iter_num)
            losses.append(loss)
            global_iters.append(global_iter)
    
    return global_iters, losses

def plot_loss_curve(global_iters, losses):
    plt.figure(figsize=(12, 6))
    plt.plot(global_iters, losses, 'b-', linewidth=1)
    plt.title('Training Loss Curve')
    plt.xlabel('Global Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 为了可读性，可以每隔一定iter显示一个epoch标记
    total_epochs = max([iter // 1251 for iter in global_iters]) + 1
    for epoch in range(total_epochs):
        iter_pos = epoch * 1251
        if iter_pos <= global_iters[-1]:
            plt.axvline(x=iter_pos, color='r', linestyle='--', alpha=0.3)
            plt.text(iter_pos, max(losses)*0.95, f'Epoch {epoch}', rotation=90, alpha=0.5)
    
    plt.tight_layout()
    output_file = os.path.join(os.path.dirname(__file__), "train_sdaa_3rd.png")
    output_file = os.path.abspath(output_file)
    plt.savefig(output_file)

# 使用示例
# log_file = 'train_sdaa_3rd.log'  # 替换为您的日志文件路径
log_file = os.path.join(os.path.dirname(__file__), "train_sdaa_3rd.log")
log_file = os.path.abspath(log_file)
global_iters, losses = parse_log_file(log_file)
plot_loss_curve(global_iters, losses)