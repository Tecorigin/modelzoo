import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
losses = []
grad_norm = []
# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        if "inf" not in parts[6]:
            loss_value = float(parts[3])
            grad_value = float(parts[6])
            losses.append(loss_value)
            grad_norm.append(grad_value)
# 将列表转换为numpy数组
losses = np.array(losses)
grad_norm = np.array(grad_norm)
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(losses.shape[0]), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.png')  # 保存Loss曲线图片

# 绘制grad norm曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(grad_norm.shape[0]), grad_norm, label='Grad Norm')
plt.xlabel('Iteration')
plt.ylabel('Grad Norm')
plt.title('Grad Norm')
plt.legend()
plt.savefig('train_grad_norm.png')  # 保存grad norm曲线图片

