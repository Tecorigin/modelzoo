import os
def get_latest_folder(directory):
    # 获取文件夹中的所有文件夹（排除文件）
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # 如果没有文件夹，则返回 None
    if not folders:
        return None
    
    # 按文件夹名排序（假设文件夹名是按时间或版本号排序的）
    latest_folder = sorted(folders)[-1]
    
    return latest_folder

# directory = '/path/to/your/folder'
# latest_folder = get_latest_folder(directory)

# if latest_folder:
#     print(f"The latest folder is: {latest_folder}")
# else:
#     print("No folders found.")

def get_latest_file(directory):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # 获取文件的修改时间，并按修改时间排序，返回最新的文件
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    
    return latest_file

directory = '../work_dirs/cascade-rcnn_r50_fpn_1x_coco'
latest_folder = get_latest_folder(directory)

print(f"The latest folder is: {latest_folder}")
latest_file = '../work_dirs/cascade-rcnn_r50_fpn_1x_coco/' + latest_folder + '/vis_data/scalars.json'
print(f"The latest file is: {latest_file}")

import json
import matplotlib.pyplot as plt

def plot_loss_from_file(file_path):
    # 初始化变量来存储不同的损失值
    iterations = []
    loss_values = []
    off_loss_values = []
    pull_loss_values = []
    push_loss_values = []

    # 读取文件并提取数据
    with open(file_path, 'r') as f:
        for line in f:
            # 每一行都是一个JSON对象
            data = json.loads(line.strip())
            
            # 提取需要的损失值和iteration
            iterations.append(data['iter'])
            loss_values.append(data['loss'])
            off_loss_values.append(data['loss_rpn_cls'])
            pull_loss_values.append(data['s0.loss_cls'])
            push_loss_values.append(data['s0.acc'])

    # 创建output文件夹（如果不存在）
    output_dir = 'output_cascade-rcnn'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 为每种损失绘制独立的图像并保存为PNG
    loss_data = [
        ('Loss', loss_values, 'blue'),
        ('loss_rpn_cls', off_loss_values, 'red'),
        ('s0.loss_cls', pull_loss_values, 'green'),
        ('s0.acc', push_loss_values, 'yellow')
    ]

    # 循环处理每个损失类型并保存图像
    for loss_name, loss_list, color in loss_data:
        plt.figure(figsize=(10, 6))
        
        # 绘制单个损失曲线
        plt.plot(iterations, loss_list, label=loss_name, color=color, marker='o')
        
        # 设置图形标题和标签
        plt.title(f'{loss_name} over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Value')
        
        # 添加图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # 保存图像为 PNG 文件到 output 文件夹
        plt.savefig(os.path.join(output_dir, f'{loss_name}_plot.png'))
        plt.close()  # 关闭当前图形，避免重叠

# 调用函数并传入文件路径
# file_path = '/path/to/your/file.txt'
plot_loss_from_file(latest_file)