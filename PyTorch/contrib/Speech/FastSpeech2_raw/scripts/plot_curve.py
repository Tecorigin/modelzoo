#绘制logs中的模型训练loss、psnr过程
import matplotlib.pyplot as plt
import numpy as np

# 初始化列表用于存储数据
save_list = {}
losses = []
step = 100
# 读取文件内容并解析数据
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        if "Total Loss:" in line:
            parts = line.strip().split(' ')
            print(parts)
            try:
                k = int(parts[1].split('/')[0])
                if k in save_list:
                    save_list[k].append(float(parts[4].split(',')[0]))
                else:
                    save_list[k] = [float(parts[4].split(',')[0])]
            except Exception as e:
                continue 
losses = []
idx = []
for k in save_list:
    l = save_list[k]
    idx.append(k//step)  
    n = len(l)
    losses.append(sum(l)/n)
print(idx)
print(losses)
srt_pairs = sorted(zip(idx,losses))    
srt_idx,srt_loss = zip(*srt_pairs)
srt_idx,srt_loss = list(srt_idx),list(srt_loss)  
'''idx = parts.index("\x1b[92mloss:")
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
print(losses.shape[0])'''
# 绘制Loss曲线并保存
plt.figure(figsize=(10, 6))
plt.plot(srt_idx, srt_loss, label='Loss')
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss.png')  # 保存Loss曲线图片