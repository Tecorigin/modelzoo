import matplotlib.pyplot as plt
import os

# 获取当前代码文件所在的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 读取文件中的loss值
loss_values = []
with open("train_sdaa_3rd.log", "r") as file:
    for line in file:
        line = line.strip()  # 去除行首行尾的空白字符
        if line:  # 如果行不为空
            try:
                loss = float(line)  # 将行内容转换为浮点数
                loss_values.append(loss)
            except ValueError:
                # 如果转换失败，跳过该行
                print(f"无法将 {line} 转换为浮点数，跳过该行。")

# 绘制loss值的图表
plt.figure(figsize=(10, 6))  # 设置图表大小
plt.plot(loss_values, marker='o')  # 使用圆圈标记每个点
plt.title("Train Loss")  # 设置图表标题
plt.xlabel("Epoch")  # 设置x轴标签
plt.ylabel("Loss")  # 设置y轴标签
plt.grid(True)  # 显示网格

# 保存图表到当前代码文件所在路径下
save_path = os.path.join(current_path, "train_loss_plot.png")
plt.savefig(save_path)
print(f"图表已保存到：{save_path}")

# 显示图表
plt.show()