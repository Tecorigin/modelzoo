import re
import matplotlib.pyplot as plt

# 日志文件路径
log_file_path = "train_sdaa_3rd.log"  # 替换为你的日志文件路径

# 用于存储提取的 Loss 值
loss_values = []

# 读取日志文件并提取 Loss 值
with open(log_file_path, "r") as file:
    for line in file:
        # 使用正则表达式匹配 Loss 值
        match = re.search(r"Loss: (\d+\.\d+)", line)
        if match:
            loss = float(match.group(1))  # 提取 Loss 值并转换为浮点数
            loss_values.append(loss)

# 绘制 Loss 值的图表
plt.figure(figsize=(12, 6))
plt.plot(loss_values, marker='o', linestyle='-', color='b', label='Loss')
plt.title("Training Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# 保存图表到当前目录
plt.savefig("training_loss_plot.png")
print("Loss plot saved as 'training_loss_plot.png' in the current directory.")

# 显示图表
plt.show()