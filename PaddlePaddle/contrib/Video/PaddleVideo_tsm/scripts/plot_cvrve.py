#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
iterations = []
losses = []

# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "top1:" in line and "top5:" in line:
            
            parts = line.strip().split(' ')
            #print(parts)
            idx = parts.index("\x1b[92mloss:")
            loss_value = float(parts[idx+1])
            #iterations.append(iter_value)
            losses.append(loss_value)

# 将列表转换为numpy数组
##iterations = np.array(iterations)
losses = np.array(losses)
plot_loss = []
interval = 5
final = len(losses)%interval
num = len(losses)//interval
for i in range(num):
    plot_loss.append(sum(losses[i*interval:(i+1)*interval])/interval)
if final:
    plot_loss.append(sum(losses[len(losses)-final:])/final)
print(losses.shape[0])
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(range(len(plot_loss)), plot_loss, label='Loss')
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片