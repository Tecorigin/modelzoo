import re
import matplotlib.pyplot as plt

def parse_line(line):
    """
    从一行日志中用正则提取:
      - epoch: 第几个epoch
      - step_in_epoch: 本epoch第几个iteration
      - total_in_epoch: 每个epoch总共有多少iteration
      - current_loss: 当前这一步输出的 loss

    返回 (global_iteration, current_loss) 或 None
    """
    pattern = r"Epoch:\s*\[(\d+)\]\[\s*(\d+)\/(\d+)\].*Loss\s+([\d\.e\+\-]+)\s*\("
    match = re.search(pattern, line)
    if match:
        epoch = int(match.group(1))
        step_in_epoch = int(match.group(2))
        total_in_epoch = int(match.group(3))
        current_loss_str = match.group(4)
        try:
            current_loss = float(current_loss_str)
        except ValueError:
            return None
        global_iteration = epoch * total_in_epoch + step_in_epoch
        return global_iteration, current_loss
    return None

def moving_average(data, window_size=5):
    """
    简单的滚动平均平滑：
    对于 data[i]，使用从 i-window_size+1 到 i 的数据做平均。
    """
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start : i + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed

def exponential_moving_average(data, alpha=0.6):
    """
    指数平滑：
    smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * data[i]
    alpha通常在(0,1)之间，越接近1表明越依赖历史平滑值，越小则对最新数据更敏感。
    """
    smoothed = []
    for i, val in enumerate(data):
        if i == 0:
            smoothed.append(val)
        else:
            prev = smoothed[-1]
            smoothed.append(alpha * prev + (1 - alpha) * val)
    return smoothed

def main(log_file, save_name):
    
    iterations = []
    losses = []
    
    # 1. 从日志文件读取数据
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            result = parse_line(line)
            if result is not None:
                it, loss_val = result
                iterations.append(it)
                losses.append(loss_val)
    
    if not iterations:
        print("未能从日志中解析到有效的迭代信息和Loss。请检查日志格式或正则表达式。")
        return

    # 2. 对Loss数据进行平滑处理
    #   你可以在此二选一，或同时对比两种结果
    window_size = 10  # 滚动窗口大小，可根据需要调整
    losses_ma = moving_average(losses, window_size=window_size)          # 滚动平均
    losses_ema = exponential_moving_average(losses, alpha=0.9)  # 指数平滑

    # 3. 绘图
    plt.figure(figsize=(8, 6))
    
    # 原始数据
    plt.plot(iterations, losses, marker='o', linestyle='--', color='gray', label='Original Loss', alpha=0.4)
    # 滚动平均
    plt.plot(iterations, losses_ma, marker='o', linestyle='-', color='blue', label=f'Moving Average (win={window_size})')
    # 指数平滑
    plt.plot(iterations, losses_ema, marker='o', linestyle='-', color='red', label='Exponential MA (alpha=0.9)')

    plt.title("Training Loss vs. Global Iteration (with smoothing)")
    plt.xlabel("Global Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 4. 保存图片到本地
    plt.savefig(save_name, dpi=200)

    # 如果需要在屏幕上查看
    # plt.show()

if __name__ == "__main__":
    num = 3
    log_file = f"../examples/imagenet/train_sdaa_3rd_log/efficientnet-b{num}_train.txt"
    save_name = f"./train_sdaa_3_log/training_loss_smoothed_{num}.png"
    main(log_file, save_name)