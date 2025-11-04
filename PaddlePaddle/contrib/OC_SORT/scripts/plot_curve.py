import matplotlib.pyplot as plt
import re

# 读取日志文件
log_file_path = "train_sdaa_3rd.log"

with open(log_file_path, "r", encoding="utf-8") as file:
    log_data = file.readlines()

# 使用正则表达式提取 loss 值
loss_pattern = re.compile(r"loss:\s*([\d.]+)")

loss_values = []
iterations = []

for idx, line in enumerate(log_data):
    match = loss_pattern.search(line)
    if match:
        loss_values.append(float(match.group(1)))
        iterations.append(idx)  # 记录对应的行号作为迭代步数

# 绘制 loss 变化曲线
plt.figure(figsize=(10, 5))
plt.plot(iterations, loss_values, label="Loss", color="blue", linewidth=1)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()

# 显示图表
plt.show()