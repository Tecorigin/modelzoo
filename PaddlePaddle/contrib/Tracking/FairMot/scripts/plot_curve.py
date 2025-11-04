#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
iterations = []
losses = []
idx = 0
# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "Loss:" in line and "INFO" in line:
            idx += 1
            #print(line)
            parts = line.strip().split(' ')
            #print(parts)
            #idx = parts.index("loss_cls:")
            
            loss_value = float(parts[-1])
            #iterations.append(iter_value)
            losses.append(loss_value)
print(idx)

# 将列表转换为numpy数组
##iterations = np.array(iterations)
losses = np.array(losses)
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(idx//2+1), losses[:idx//2+1], label='Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片