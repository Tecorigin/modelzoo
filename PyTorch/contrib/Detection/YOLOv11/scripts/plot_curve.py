import re
import matplotlib.pyplot as plt

# 定义正则表达式来提取 box_loss、cls_loss 和 dfl_loss
loss_pattern = re.compile(r'\d+/\d+\s+\d+\.\d+G\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)')

# 存储提取的损失值
box_losses = []
cls_losses = []
dfl_losses = []

# 读取日志文件
with open('train_sdaa_3rd.log', 'r') as file:
    for line in file:
        # 匹配损失值
        match = loss_pattern.search(line)
        if match:
            box_losses.append(float(match.group(1)))
            cls_losses.append(float(match.group(2)))
            dfl_losses.append(float(match.group(3)))

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(box_losses, label='Box Loss', color='blue', linestyle='-')
plt.plot(cls_losses, label='Cls Loss', color='red', linestyle='--')
plt.plot(dfl_losses, label='Dfl Loss', color='green', linestyle=':')

# 添加标题和标签
plt.title('Training Loss Curves')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('train_sdaa_3rd.png')