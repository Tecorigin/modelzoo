#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据

losses = []
iter_list = []
step = 100
# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "Loss" in line:
            parts = line.strip().split(' ')
            losses.append(float(parts[4]))
            iter_list.append(int(parts[2]))
            
            
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(iter_list, losses, label='Loss')
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片