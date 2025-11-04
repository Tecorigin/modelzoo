import csv
import torchaudio
import os
import random
# 定义 CSV 文件路径和新的 CSV 文件路径
csv_file = '/data/datasets/LJSpeech-1.1/metadata.csv'
new_csv_file = 'new_metadata.csv'

'''# 定义表头
fieldnames = ['audio_path', 'text', 'duration']

# 打开原始 CSV 文件和新的 CSV 文件
with open(csv_file, 'r', encoding='utf-8') as infile, open(new_csv_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    # 写入表头
    writer.writeheader()

    # 遍历原始 CSV 文件的每一行
    for row in reader:
        print("===",row)
        audio_path = row[0]
        text = row[1]
        # 检查音频文件是否存在
        if os.path.exists(audio_path):
            try:
                # 使用 torchaudio 加载音频文件
                waveform, sample_rate = torchaudio.load(audio_path)
                # 计算音频时长（秒）
                duration = waveform.size(1) / sample_rate
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                duration = None
        else:
            print(f"File {audio_path} not found.")
            duration = None
        # 创建包含所有信息的字典
        new_row = {
            'audio_path': audio_path,
            'text': text,
            'duration': duration
        }
        # 写入包含时长信息的行到新的 CSV 文件
        writer.writerow(new_row)

print(f"Processed data saved to {new_csv_file}")'''
'''csv_file = '/data/datasets/22001/LJSpeech-1.1/metadata.csv'

with open(csv_file, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除行尾的换行符
        line = line.strip()
        new_line = line.split('|')
        path = new_line[0]
        text = new_line[1]
        
        break'''
        
import csv
import torchaudio
import os

# 定义输入和输出 CSV 文件的路径
csv_file = '/data/datasets/LJSpeech-1.1/metadata.csv'
train_csv_file = 'train.csv'
test_csv_file = "test.csv"

# 定义新 CSV 文件的表头
fieldnames = ['audio_path', 'text', 'duration']

# 打开输入和输出文件
with open(csv_file, 'r', encoding='utf-8') as infile, open(train_csv_file, 'w', newline='', encoding='utf-8') as outfile,open(test_csv_file, 'w', newline='', encoding='utf-8') as outfile2:
    # 创建 CSV 写入器并写入表头
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer2 = csv.DictWriter(outfile2, fieldnames=fieldnames)
    writer2.writeheader()

    # 逐行读取输入文件
    for line in infile:
        # 去除行尾的换行符
        line = line.strip()
        # 使用竖线分割每行内容
        new_line = line.split('|')
        # 提取路径和文本
        path = new_line[0]
        text = new_line[1]
        audio_path = os.path.join("/data/datasets/LJSpeech-1.1/wavs",path)+'.wav'
        # 检查音频文件是否存在
        if os.path.exists(audio_path):
            try:
                # 使用 torchaudio 加载音频文件
                waveform, sample_rate = torchaudio.load(audio_path)
                # 计算音频时长（秒）
                duration = waveform.size(1) / sample_rate
                duration = round(duration, 1)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                duration = None
        else:
            print(f"File {audio_path} not found.")
            duration = None

        # 创建包含所有信息的字典
        new_row = {
            'audio_path': audio_path,
            'text': text,
            'duration': duration
        }
        # 将新行写入输出的 CSV 文件
        if random.random() > 0.9:
            writer2.writerow(new_row)
        else:
            writer.writerow(new_row)
