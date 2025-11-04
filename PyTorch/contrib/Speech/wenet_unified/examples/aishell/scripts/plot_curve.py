import yaml
import matplotlib.pyplot as plt
import numpy as np



import os
losses = []
acc_list = []

with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        acc_value = float(parts[0].split(':')[1])
        loss_value = float(parts[1].split(':')[1])
        #iterations.append(iter_value)
        losses.append(loss_value)
        acc_list.append(acc_value)
losses = np.array(losses)
acc_list = np.array(acc_list)

plt.figure(figsize=(10, 6))
plt.plot(range(losses.shape[0]), losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.png')  # 保存Loss曲线图片

plt.figure(figsize=(10, 6))
plt.plot(range(acc_list.shape[0]), acc_list, label='acc')
plt.xlabel('Iteration')
plt.ylabel('acc')
plt.title('acc')
plt.legend()
plt.savefig('train_acc.png')  # 保存grad norm曲线图片

