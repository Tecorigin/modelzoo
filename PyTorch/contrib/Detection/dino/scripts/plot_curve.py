import re
import matplotlib.pyplot as plt

# 读取文本文件
file_path = 'train_3rd_sdaa.log'  # 替换为实际的文件路径
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"未找到文件: {file_path}")
else:
    # 提取 epoch 和 train_loss
    epoch_pattern = r'"epoch": (\d+)'
    loss_pattern = r'"train_loss": ([\d.]+)'
    epochs = [int(match.group(1)) for match in re.finditer(epoch_pattern, content)]
    losses = [float(match.group(1)) for match in re.finditer(loss_pattern, content)]

    # 绘制曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.xticks(rotation=45)
    plt.ylabel('Train Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_sdaa_3rd.png')