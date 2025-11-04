import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
# Adapted to tecorigin hardware
# 日志文件路径
file_path = '/data/modelzoo/PyTorch/contrib/Classification/ACmix/experiments/train_sdaa_3rd.log/train_sdaa_3rd.log'

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

# 获取起始时间和2小时后的结束时间
try:
    start_time = datetime.fromtimestamp(float(log_data[0]['timestamp']))
    end_time = start_time + timedelta(hours=2)
except KeyError:
    print("The first entry in the log does not contain a 'timestamp' field.")
    exit(1)

# 提取需要的数据
timestamps = []
train_losses = []
val_accs = []

for entry in log_data:
    if entry.get('type') == 'LOG' and 'data' in entry:
        try:
            data = entry['data']  # 这里的 data 已是合法的字典
            timestamp = datetime.fromtimestamp(float(entry['timestamp']))
            
            if timestamp > end_time:
                break
            
            # 提取损失值
            if 'train.loss' in data:
                train_losses.append(float(data['train.loss']))
                timestamps.append(timestamp)
            
            # 提取准确率值
            if 'val.acc' in data:
                val_accs.append(float(data['val.acc']))
        except (KeyError, ValueError) as e:
            print(f"Error processing entry: {entry}, error: {e}")

# 检查是否有有效数据
if not timestamps or not train_losses:
    print("No valid training data found in the log.")
    exit(1)

# 创建 DataFrame 并平滑数据
data = pd.DataFrame({"timestamps": timestamps, "train_losses": train_losses})
data['smoothed_loss'] = data['train_losses'].rolling(window=50).mean()  # 移动平均窗口为50

# 如果准确率存在，也创建 DataFrame 并平滑
if val_accs:
    acc_data = pd.DataFrame({"timestamps": timestamps[:len(val_accs)], "val_accs": val_accs})
    acc_data['smoothed_acc'] = acc_data['val_accs'].rolling(window=50).mean()

# 绘制损失图
plt.figure(figsize=(16, 8))

# 原始数据曲线
plt.plot(data['timestamps'], data['train_losses'], alpha=0.3, label='Original Train Loss', color='red')

# 平滑数据曲线
plt.plot(data['timestamps'], data['smoothed_loss'], label='Smoothed Train Loss', color='blue')

# 设置标题和坐标轴
plt.title('Train Loss over Time (First 2 Hours - Smoothed)', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(rotation=45, fontsize=10)  # 旋转时间刻度以避免重叠

# 添加图例
plt.legend(fontsize=12)

# 保存损失图像
plt.tight_layout()
plt.savefig('/data/modelzoo/PyTorch/contrib/Classification/ACmix/scripts/train_sdaa_3rd.png')
plt.show()

# 绘制准确率图（如果存在）
if val_accs:
    plt.figure(figsize=(16, 8))

    # 原始数据曲线
    plt.plot(acc_data['timestamps'], acc_data['val_accs'], alpha=0.3, label='Original Validation Accuracy', color='green')

    # 平滑数据曲线
    plt.plot(acc_data['timestamps'], acc_data['smoothed_acc'], label='Smoothed Validation Accuracy', color='orange')

    # 设置标题和坐标轴
    plt.title('Validation Accuracy over Time (First 2 Hours - Smoothed)', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45, fontsize=10)  # 旋转时间刻度以避免重叠

    # 添加图例
    plt.legend(fontsize=12)

    # 保存准确率图像
    plt.tight_layout()
    plt.savefig('/data/modelzoo/PyTorch/contrib/Classification/ACmix/scripts/train_sdaa_3rd.png')
    plt.show()
