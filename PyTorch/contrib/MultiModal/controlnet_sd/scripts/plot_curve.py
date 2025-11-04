import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
losses = []
# 读取文件内容并解析数据
idx = 0
loss_sum = 0
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "Steps" in line and "loss" in line:
            parts = line.strip().split(' ')
            for item in parts:
                if "loss=" in item:
                    idx += 1
                    loss_value = float(item.split("=")[-1].split(',')[0])
                    if idx % 5 ==0:
                        losses.append(loss_sum/5)
                        loss_sum = 0 
                    loss_sum += loss_value

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

