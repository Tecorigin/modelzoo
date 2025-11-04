import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
losses = []

# 读取文件内容并解析数据
idx = 0
v_sum = 0
with open('model_train_log_rank0.txt', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        print(parts)
        if "Loss" in parts[0]:
            idx += 1
            loss_value = float(parts[1])
            v_sum += loss_value
            if idx == 50:
                losses.append(v_sum/idx)
                idx = 0
                v_sum = 0

# 将列表转换为numpy数组
losses = np.array(losses)

# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(losses.shape
               [0]), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_sdaa_3rd.png')  # 保存Loss曲线图片

