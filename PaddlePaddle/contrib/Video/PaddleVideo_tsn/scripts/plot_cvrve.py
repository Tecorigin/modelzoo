#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
iterations = []
losses = []

# 读取文件内容并解析数据
isok = 0
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "epoch:" in line and "top1:" in line and isok%2 == 0:
            
            parts = line.strip().split(' ')
            idx = parts.index("\x1b[92mloss:")
            loss_value = float(parts[idx+1])
            #iterations.append(iter_value)
            losses.append(loss_value)
        isok += 1

# 将列表转换为numpy数组
##iterations = np.array(iterations)
losses = np.array(losses)
print(losses.shape[0])
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(losses.shape[0]), losses, label='Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片