import re
import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数来读取日志文件并提取相关数据
def parse_log_file(log_filename):
    all_iterations = []
    all_losses = []
    seen_iterations = set()  # 用于存储已经处理过的iteration

    with open(log_filename, 'r') as f:
        lines = f.readlines()

    # 使用正则表达式从每一行中提取信息
    pattern = re.compile(r"Epoch: \[(\d+)\]\[(\d+/\d+)\].*Loss (\d+\.\d+)")

    for line in lines:
        matches = pattern.findall(line)  # 使用findall来提取一行中的所有匹配项
        for match in matches:
            iteration = match[1]  # 例如: 100/1252
            loss = float(match[2])  # 获取Loss值
            
            iter_number = int(iteration.split('/')[0])  # 提取迭代次数
            
            if iter_number not in seen_iterations:  # 如果这个iteration没出现过
                all_iterations.append(iter_number)
                all_losses.append(loss)
                seen_iterations.add(iter_number)  # 标记这个iteration已处理过

    return np.array(all_iterations), np.array(all_losses)

# 绘制损失随迭代变化的图
def plot_loss(iterations, losses):
    plt.plot(iterations, losses, label="Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("train_sdaa_3rd.png")
    plt.show()

# 主程序
log_filename = 'train_sdaa_3rd.log'  # 替换为你的日志文件路径
iterations, losses = parse_log_file(log_filename)
plot_loss(iterations, losses)
