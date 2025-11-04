import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
losses = []

# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        loss_value = float(parts[1].split(":")[1])
        losses.append(loss_value)

# 将列表转换为numpy数组
losses = np.array(losses)
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(losses.shape[0]), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.png')  # 保存Loss曲线图片

