import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from argparse import ArgumentParser

def parse_log_step_loss(string):
    # 解析出每行中的 step 和 loss
    pattern = r"loss:\s*([\d\.eE+-]+)\s+-.*step:\s*(\d+)"
    step_loss = []
    for line in string.splitlines():
        match = re.search(pattern, line)
        if match:
            loss = float(match.group(1))
            step = int(match.group(2))
            step_loss.append((step, loss))
    return dict(step_loss)
def compare_loss(benchmark_loss_array, sdaa_loss_array):
    def MeanRelativeError(cuda_loss, sdaa_loss):
        return np.mean((sdaa_loss - cuda_loss) / cuda_loss)

    def MeanAbsoluteError(cuda_loss, sdaa_loss):
        return np.mean(sdaa_loss - cuda_loss)

    benchmark_compare_loss = benchmark_loss_array
    sdaa_compare_loss = sdaa_loss_array

    mean_relative_error = MeanRelativeError(benchmark_compare_loss, sdaa_compare_loss)
    mean_absolute_error = MeanAbsoluteError(benchmark_compare_loss, sdaa_compare_loss)

    print("MeanRelativeError:", mean_relative_error)
    print("MeanAbsoluteError:", mean_absolute_error)

    print_str = f"{mean_relative_error=} <= 0.05 or {mean_absolute_error=} <= 0.0002"
    if mean_relative_error <= 0.05 or mean_absolute_error <= 0.0002:
        print("pass", print_str)
        return True, print_str
    else:
        print("fail", print_str)
        return False, print_str

def plot_loss(steps, sdaa_loss, cuda_loss):
    fig, ax = plt.subplots(figsize=(12, 6))

    smoothed_sdaa = savgol_filter(sdaa_loss, 5, 1)
    smoothed_cuda = savgol_filter(cuda_loss, 5, 1)

    ax.plot(steps, smoothed_sdaa, label="sdaa_loss", linewidth=2)
    ax.plot(steps, smoothed_cuda, "--", label="cuda_loss", linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves (Smoothed)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("loss.jpg")
    print("Saved plot to loss.jpg")

if __name__ == "__main__":
    parser = ArgumentParser(description='Compare loss from two logs')
    parser.add_argument('--sdaa-log', type=str, default="sdaa.log")
    parser.add_argument('--cuda-log', type=str, default="cuda.log")
    args = parser.parse_args()

    # 读取 log 文件
    with open(args.sdaa_log, 'r') as f:
        sdaa_log_str = f.read()
    with open(args.cuda_log, 'r') as f:
        cuda_log_str = f.read()

    # 解析 step-loss 映射
    sdaa_dict = parse_log_step_loss(sdaa_log_str)
    cuda_dict = parse_log_step_loss(cuda_log_str)

    # 提取交集的 step
    common_steps = sorted(set(sdaa_dict.keys()) & set(cuda_dict.keys()))
    if not common_steps:
        print("No common steps found between the two logs.")
        exit(1)

    sdaa_loss = np.array([sdaa_dict[step] for step in common_steps])
    cuda_loss = np.array([cuda_dict[step] for step in common_steps])

    # 比较 loss
    compare_loss(cuda_loss, sdaa_loss)

    # 画图
    plot_loss(common_steps, sdaa_loss, cuda_loss)
