import re
import matplotlib.pyplot as plt

def extract_iter_and_loss_from_log(file_path):
    iter_values = []
    loss_values = []
    try:
        # 打开日志文件并读取内容
        with open(file_path, 'r') as file:
            log_content = file.read()
            # 使用正则表达式匹配 iter 和 loss 的值
            pattern =  r"Epoch\(train\)  \[(\d+)\]\[\d+/\d+\].*?loss: (\d+\.\d+)"
            matches = re.findall(pattern, log_content)
            for match in matches:
                iter_values.append(int(match[0]))
                loss_values.append(float(match[1]))
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    return iter_values, loss_values

def extract_iter_and_acc1_from_log(file_path):
    iter_values = []
    loss_values = []
    try:
        # 打开日志文件并读取内容
        with open(file_path, 'r') as file:
            log_content = file.read()
            # 使用正则表达式匹配 iter 和 loss 的值
            pattern =   r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?accuracy/top1: (\d+\.\d+)"
            matches = re.findall(pattern, log_content)
            for match in matches:
                iter_values.append(int(match[0]))
                loss_values.append(float(match[1]))
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    return iter_values, loss_values


def extract_iter_and_acc5_from_log(file_path):
    iter_values = []
    loss_values = []
    try:
        # 打开日志文件并读取内容
        with open(file_path, 'r') as file:
            log_content = file.read()
            # 使用正则表达式匹配 iter 和 loss 的值
            pattern =  r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?accuracy/top5: (\d+\.\d+)"
            matches = re.findall(pattern, log_content)
            for match in matches:
                iter_values.append(int(match[0]))
                loss_values.append(float(match[1]))
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    return iter_values, loss_values

def plot_loss_curve(iter_values, loss_values):
    # 绘制 loss 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_values, loss_values)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('train_sdaa_3rd.png')

def plot_acc1_curve(iter_values, loss_values):
    # 绘制 loss 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_values, loss_values)
    plt.title('Acc/top1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('acc/top1')
    plt.grid(True)
    plt.savefig('acctop1.png')

def plot_acc5_curve(iter_values, loss_values):
    # 绘制 loss 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_values, loss_values)
    plt.title('Acc/top5 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('acc/top5')
    plt.grid(True)
    plt.savefig('acctop5.png')

if __name__ == "__main__":
    # 请替换为你的日志文件路径
    log_file_path = 'train_sdaa_3rd.log'
    iter_values, loss_values = extract_iter_and_loss_from_log(log_file_path)
    if iter_values and loss_values:
        plot_loss_curve(iter_values, loss_values)

    iter_values, acc1_values = extract_iter_and_acc1_from_log(log_file_path)
    if iter_values and acc1_values:
        plot_acc1_curve(iter_values, acc1_values)

    iter_values, acc5_values = extract_iter_and_acc5_from_log(log_file_path)
    if iter_values and acc5_values:
        plot_acc5_curve(iter_values, acc5_values)