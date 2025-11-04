import matplotlib.pyplot as plt
import numpy as np
def plot_cur(log_name):
    # 初始化列表用于存储数据
    losses = []
    acc_list = []
    name = log_name.split(".")[0].split("_")[-1]
    # 读取文件内容并解析数据
    with open("./scripts/"+log_name+".log", 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            print(parts[3])
            loss_value = float(parts[3])
            acc_value = float(parts[6])
            #print(loss_value)
            losses.append(loss_value)
            acc_list.append(acc_value)
    # 将列表转换为numpy数组
    losses = np.array(losses)
    acc_list = np.array(acc_list)
    # 绘制Loss曲线并保存
    plt.figure(figsize=(10, 6))
    plt.plot(range(losses.shape[0]), losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss-cora')
    plt.legend()
    plt.savefig(f'./script/train_sdaa_3rd_loss_{name}.png')  # 保存Loss曲线图片

    plt.figure(figsize=(10, 6))
    plt.plot(range(acc_list.shape[0]), acc_list, label='acc')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.title('Training acc-citeseer')
    plt.legend()
    plt.savefig(f'./script/train_sdaa_3rd_acc_{name}.png')  # 保存acc曲线图片
                
                
log_folder = ["train_model_log_cora","train_model_log_citeseer","train_model_log_pubmed"]

for log_name in log_folder:
    plot_cur(log_name)