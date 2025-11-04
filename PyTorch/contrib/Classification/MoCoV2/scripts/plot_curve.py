import matplotlib.pyplot as plt
import re
import os

# 日志文件路径
log_file_path = '../experiments/train_sdaa_3rd/train_sdaa_3rd.log'

# 用来存储每个epoch的损失值
epoch_losses = []

# 打开日志文件并读取内容
with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()

    # 正则表达式匹配每个 epoch 中的 loss 值
    for line in lines:
        match = re.search(r'\[Train\] loss: (\d+\.\d+)', line)
        if match:
            # 将每个匹配到的损失值添加到 epoch_losses 列表
            epoch_losses.append(float(match.group(1)))

# 创建一个 epoch 数值列表（从 1 到 n_epoch）
epochs = list(range(1, len(epoch_losses) + 1))

# 绘制训练损失曲线
plt.plot(epochs, epoch_losses, label='Pretrain Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve (Pretrain)')
plt.grid(True)
plt.legend()

# 保存图表到指定路径
save_path = '../scripts/train_sdaa_3rd.png'
plt.savefig(save_path)

# 显示图表
plt.show()

print(f"Training loss curve saved to {save_path}")
