#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
iterations = []
losses = []
psnrs = []

# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        
        parts = line.strip().split(' ')
        print(parts)
        iter_value = int(parts[2])
        loss_value = float(parts[4])
        psnr_value = float(parts[7])
        iterations.append(iter_value)
        losses.append(loss_value)
        psnrs.append(psnr_value)

# 将列表转换为numpy数组
iterations = np.array(iterations)
losses = np.array(losses)
psnrs = np.array(psnrs)

# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片

# 绘制PSNR曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(iterations, psnrs, label='PSNR')
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.title('Training PSNR')
plt.legend()
plt.savefig('training_psnr.png')  # 保存PSNR曲线图片