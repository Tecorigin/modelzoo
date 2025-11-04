import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# 日志文件路径
file_path = '../experiments/train_sdaa_3rd.log/train_sdaa_3rd.log'

# 尝试读取日志文件
log_data = []
try:
    with open(file_path, 'r') as f:
        for line in f:
            try:
                # 去掉前缀 "TCAPPDLL " 或 "INFO " 并解析 JSON
                if line.startswith("TCAPPDLL "):
                    line = line[len("TCAPPDLL "):]  # 去掉前缀
                elif line.startswith("INFO "):
                    line = line[len("INFO "):]  # 去掉前缀

                log_entry = json.loads(line)  # 尝试解析 JSON
                
                # 如果是 INFO 类型的日志，跳过
                if log_entry.get('type') == 'INFO':
                    continue

                # 修复并解析 data 字段（如果存在）
                if 'data' in log_entry and isinstance(log_entry['data'], str):
                    log_entry['data'] = json.loads(
                        log_entry['data']
                        .replace("'", "\"")  # 替换单引号为双引号
                        .replace("None", "null")  # 替换 None 为 null
                    )
                
                log_data.append(log_entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line.strip()}, error: {e}")
except FileNotFoundError:
    print(f"Log file not found: {file_path}")
    exit(1)

# 检查是否读取到数据
if not log_data:
    print("Log file is empty or no valid entries were parsed.")
    exit(1)

# 提取需要的数据
batch_numbers = []  # 记录批次号
train_losses = []  # 记录训练损失

for i, entry in enumerate(log_data):
    if entry.get('type') == 'LOG' and 'data' in entry:
        try:
            data = entry['data']  # 这里的 data 已是合法的字典
            
            # 提取损失值
            if 'train.loss' in data:
                train_losses.append(float(data['train.loss']))
                batch_numbers.append(i + 1)  # i+1 表示第几个批次
        except (KeyError, ValueError) as e:
            print(f"Error processing entry: {entry}, error: {e}")

# 检查是否有有效数据
if not batch_numbers or not train_losses:
    print("No valid training data found in the log.")
    exit(1)

# 创建 DataFrame
data = pd.DataFrame({"batch_numbers": batch_numbers, "train_losses": train_losses})

# 绘制损失图
plt.figure(figsize=(16, 8))

# 原始数据曲线
plt.plot(data['batch_numbers'], data['train_losses'], label='Train Loss', color='red')

# 设置标题和坐标轴
plt.title('Train Loss over Batches', fontsize=16)
plt.xlabel('Batch Number', fontsize=14)
plt.ylabel('Loss', fontsize=14)

# 添加图例
plt.legend(fontsize=12)

# 保存损失图像
plt.tight_layout()
plt.savefig('../scripts/train_sdaa_3rd.png')
plt.show()
