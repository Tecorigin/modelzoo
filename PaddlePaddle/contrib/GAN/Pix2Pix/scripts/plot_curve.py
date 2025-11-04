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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取日志文件
with open('./output_dir/pix2pix_facades-2025-01-25-07-47/log.txt', 'r') as file:
    log_data = file.readlines()

# 正则表达式匹配训练数据
pattern = re.compile(
    r'Epoch: (\d+)/\d+, iter: \d+/\d+ lr: [\d\.e+-]+ '
    r'D_fake_loss: ([\d\.]+) D_real_loss: ([\d\.]+) '
    r'G_adv_loss: ([\d\.]+) G_L1_loss: ([\d\.]+)'
)

epoch_data = {}  # 用于存储每轮的最后一个数据点

for line in log_data:
    match = pattern.search(line)
    if match:
        epoch = int(match.group(1))
        d_fake_loss = float(match.group(2))
        d_real_loss = float(match.group(3))
        g_adv_loss = float(match.group(4))
        g_l1_loss = float(match.group(5))

        # 更新当前 epoch 的数据
        epoch_data[epoch] = {
            'D_fake_loss': d_fake_loss,  # 判别器对生成图像的损失
            'D_real_loss': d_real_loss,  # 判别器对真实图像的损失
            'G_adv_loss': g_adv_loss,    # 生成器的对抗损失
            'G_L1_loss': g_l1_loss,      # 生成器的 L1 损失
        }

# 转换为 DataFrame
df = pd.DataFrame.from_dict(epoch_data, orient='index')
df.index.name = 'Epoch'
df.reset_index(inplace=True)

# 处理 G_L1_loss 中的 NaN 和异常极大值
df['G_L1_loss_processed'] = df['G_L1_loss'].apply(
    lambda x: 0 if np.isnan(x) else (200 if x > 200 else x)
)

# 保存到 CSV 文件
df.to_csv('./output_dir/training_losses.csv', index=False)

# 绘制损失曲线
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(df['Epoch'], df['D_fake_loss'], label='D_fake_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('D_fake_loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(df['Epoch'], df['D_real_loss'], label='D_real_loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('D_real_loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(df['Epoch'], df['G_adv_loss'], label='G_adv_loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('G_adv_loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(df['Epoch'], df['G_L1_loss_processed'], label='G_L1_loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('G_L1_loss (NaN -> 0, >200 -> 200)')
plt.legend()

plt.tight_layout()
plt.savefig('./output_dir/training_losses.png')  # 保存图像
plt.savefig('./training_losses.png')  # 保存图像
plt.show()