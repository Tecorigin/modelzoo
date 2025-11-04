import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def compare_loss(benchmark_loss_array, sdaa_loss_array):
    if len(benchmark_loss_array) == 0 or len(sdaa_loss_array) == 0:
        print("Error: Empty loss arrays, cannot compute errors")
        return False, "Empty loss arrays"
    if len(benchmark_loss_array) != len(sdaa_loss_array):
        print(f"Warning: Loss arrays have different lengths ({len(benchmark_loss_array)} vs {len(sdaa_loss_array)})")

    def MeanRelativeError(cuda_loss, sdaa_loss):
        mask = cuda_loss != 0  # 防止除以零
        if not mask.any():
            return np.inf
        return ((sdaa_loss[mask] - cuda_loss[mask]) / cuda_loss[mask]).mean()

    def MeanAbsoluteError(cuda_loss, sdaa_loss):
        return (sdaa_loss - cuda_loss).mean()

    mean_relative_error = MeanRelativeError(benchmark_loss_array, sdaa_loss_array)
    mean_absolute_error = MeanAbsoluteError(benchmark_loss_array, sdaa_loss_array)

    print("MeanRelativeError:", mean_relative_error)
    print("MeanAbsoluteError:", mean_absolute_error)

    print_str = f"{mean_relative_error=} <= 0.05 or {mean_absolute_error=} <= 0.0002"
    if mean_relative_error <= 0.05 or mean_absolute_error <= 0.0002:
        print('pass', print_str)
        return True, print_str
    else:
        print('fail', print_str)
        return False, print_str


def parse_string(string):
    # 支持两种日志格式的正则表达式
    pattern = r"rank\s*:\s*0\s+train\.loss\s*:\s*(\d+\.\d+[eE]?[-+]?\d*)"
    matches = re.findall(pattern, string)
    if not matches:
        print("Warning: No matches for rank 0 pattern, trying alternative pattern")
        pattern1 = r"loss\s+(\d+\.\d+[eE]?[-+]?\d*)"
        matches = re.findall(pattern1, string)
    if not matches:
        print("Error: No valid loss values found in log")
        return []
    print("Parsed losses:", matches)
    return matches


def parse_loss(ret_list):
    if not ret_list:
        print("Error: No loss values to parse")
        return np.array([])
    try:
        loss_arr = np.array(ret_list, dtype=float)
    except ValueError as e:
        print(f"Error: Failed to convert loss values to float: {e}")
        return np.array([])
    print("Loss array:", loss_arr)
    return loss_arr


def plot_loss(sdaa_loss, a100_loss):
    if len(sdaa_loss) == 0 or len(a100_loss) == 0:
        print("Error: Cannot plot empty loss arrays")
        return
    if len(sdaa_loss) != len(a100_loss):
        print(f"Warning: Loss arrays have different lengths ({len(sdaa_loss)} vs {len(a100_loss)})")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(sdaa_loss)))

    # 根据数组长度决定是否平滑
    if len(sdaa_loss) >= 5:
        smoothed_sdaa = savgol_filter(sdaa_loss, 5, 1)
    else:
        print("Warning: SDAA loss array too short for smoothing, using raw data")
        smoothed_sdaa = sdaa_loss
    ax.plot(x, smoothed_sdaa, label="sdaa_loss")

    if len(a100_loss) >= 5:
        smoothed_a100 = savgol_filter(a100_loss, 5, 1)
    else:
        print("Warning: CUDA loss array too short for smoothing, using raw data")
        smoothed_a100 = a100_loss
    ax.plot(x, smoothed_a100, "--", label="cuda_loss")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves")
    ax.legend()
    plt.savefig("loss.jpg")
    plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Model Zoo Loss Comparison')
    parser.add_argument('--sdaa-log', type=str, default="sdaa.log")
    parser.add_argument('--cuda-log', type=str, default="cuda.log")
    args = parser.parse_args()

    try:
        with open(args.sdaa_log, 'r') as f:
            sdaa_res = parse_string(f.read())
    except FileNotFoundError:
        print(f"Error: SDAA log file {args.sdaa_log} not found")
        exit(1)

    try:
        with open(args.cuda_log, 'r') as f:
            a100_res = parse_string(f.read())
    except FileNotFoundError:
        print(f"Error: CUDA log file {args.cuda_log} not found")
        exit(1)

    length = min(len(a100_res), len(sdaa_res))
    if length == 0:
        print("Error: No valid loss values parsed from logs")
        exit(1)
    if len(a100_res) != len(sdaa_res):
        print(f"Warning: Log lengths differ (SDAA: {len(sdaa_res)}, CUDA: {len(a100_res)}), using {length}")

    sdaa_loss = parse_loss(sdaa_res[:length])
    a100_loss = parse_loss(a100_res[:length])
    compare_loss(a100_loss, sdaa_loss)
    plot_loss(sdaa_loss, a100_loss)