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
    从 train_sdaa_3rd.log 中解析每个 Epoch 的平均 Loss（多个 Rank 的 Loss 做平均）
    以及 LFWACC（在日志中对应 'LFWACC=...'）。
    返回三个列表：epochs, mean_losses, accs
      - epochs: [1, 2, 3, ...]
      - mean_losses: 与 epochs 一一对应的平均 Loss
      - accs: 与 epochs 一一对应的 LFWACC
    """
    # 用于存储解析结果
    epochs = []  # 每个 epoch 的编号
    mean_losses = []  # 每个 epoch 的平均 loss
    acc_list = []  # 每个 epoch 的 LFWACC

    # 临时存储：当前正在处理的 epoch 以及其所有 loss
    current_epoch = None
    epoch_loss_sum = 0.0
    epoch_loss_count = 0

    # 用于在读取日志时，先按 (epoch -> LFWACC) 存起来，最后再和 epochs 对齐
    epoch2acc = {}

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1) 检查是否是 "Epoch X start training" 的行，表明下一个 epoch 开始了
            match_epoch_start = re.search(r'Epoch\s+(\d+)\s+start training', line)
            if match_epoch_start:
                # 如果已有旧的 epoch，先把它的平均 loss 记录下来
                if (current_epoch is not None) and (epoch_loss_count > 0):
                    avg_loss = epoch_loss_sum / epoch_loss_count
                    epochs.append(current_epoch)
                    mean_losses.append(avg_loss)

                # 更新为新的 epoch
                current_epoch = int(match_epoch_start.group(1))
                epoch_loss_sum = 0.0
                epoch_loss_count = 0
                continue

            # 2) 匹配训练时的 loss 记录行，例如：
            #    [Rank 1] Train Epoch: 1 [102400/489246 (84%)]200, Loss: 20.612793, ...
            match_loss = re.search(r'Train Epoch:\s*(\d+).*?Loss:\s*([\d.]+)', line)
            if match_loss:
                epoch_in_line = int(match_loss.group(1))
                loss_val = float(match_loss.group(2))

                # 只记录当前 epoch 的 loss（避免日志里重复的或其他 epoch 的混淆）
                if current_epoch == epoch_in_line:
                    epoch_loss_sum += loss_val
                    epoch_loss_count += 1
                continue

            # 3) 匹配 LFWACC=..., 例如： LFWACC=0.7778 std=...
            match_acc = re.search(r'LFWACC=(\d+\.\d+)', line)
            if match_acc:
                if current_epoch is not None:
                    acc_val = float(match_acc.group(1))
                    # 先存到字典中，后续再和 epochs 对齐
                    epoch2acc[current_epoch] = acc_val
                continue

    # 文件读完后，如果最后一个 epoch 数据没有写入，则补写
    if (current_epoch is not None) and (epoch_loss_count > 0):
        avg_loss = epoch_loss_sum / epoch_loss_count
        epochs.append(current_epoch)
        mean_losses.append(avg_loss)

    # 根据 epochs 列表，将 acc_list 对齐
    for ep in epochs:
        if ep in epoch2acc:
            acc_list.append(epoch2acc[ep])
        else:
            # 万一某个 epoch 找不到对应的 ACC，就用 None 或者 0 之类占位
            acc_list.append(None)

    return epochs, mean_losses, acc_list


def plot_loss_and_acc(epochs, losses, accs, save_fig):
    """
    使用双 y 轴绘制 Loss 和 LFWACC 与 Epoch 的关系
    """
    fig, ax1 = plt.subplots()

    # 左 y 轴：Loss
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss)
    l1 = ax1.plot(epochs, losses, '-o', color=color_loss, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # 右 y 轴：LFW-ACC
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('LFW-ACC', color=color_acc)
    l2 = ax2.plot(epochs, accs, '-s', color=color_acc, label='LFW-ACC')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # 合并图例
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='best')
    plt.title('Training Loss & LFW-ACC over Epochs')

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.show()


if __name__ == '__main__':
    # 修改为你的日志文件路径
    log_file_path = 'train_sdaa_3rd.log'
    save_fig_path = 'train_sdaa_3rd.png'

    # 解析日志，获得 epoch, loss, acc
    epochs, mean_losses, acc_list = parse_log_file(log_file_path)

    # 绘图
    plot_loss_and_acc(epochs, mean_losses, acc_list, save_fig_path)
