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
            pattern = r"Loss at (\d+) mini-batch: (\d+\.\d+.*)"
            matches = re.findall(pattern, log_content)
            iters = 0
            for match in matches:
                iter_values.append(iters)
                loss_values.append(float(match[1]))
                iters += 1
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
    return iter_values, loss_values

def plot_loss_curve(iter_values, loss_values):
    # 绘制 loss 曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(iter_values, loss_values)
    plt.title('Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('train_sdaa_3rd.png')

if __name__ == "__main__":
    # 请替换为你的日志文件路径
    log_file_path = 'train_sdaa_3rd.log'
    iter_values, loss_values = extract_iter_and_loss_from_log(log_file_path)
    if iter_values and loss_values:
        plot_loss_curve(iter_values, loss_values)
 