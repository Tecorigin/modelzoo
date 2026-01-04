import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from scipy.signal import savgol_filter
def compare_loss(benchmark_loss_array, sdaa_loss_array):

    def MeanRelativeError(cuda_loss, sdaa_loss):
        return ((sdaa_loss - cuda_loss) / cuda_loss).mean()

    def MeanAbsoluteError(cuda_loss, sdaa_loss):
        return (sdaa_loss - cuda_loss).mean()

    benchmark_mean_loss = benchmark_loss_array
    sdaa_mean_loss = sdaa_loss_array

    benchmark_compare_loss = benchmark_mean_loss
    sdaa_compare_loss = sdaa_mean_loss
    mean_relative_error = MeanRelativeError(benchmark_compare_loss, sdaa_compare_loss)
    mean_absolute_error = MeanAbsoluteError(benchmark_compare_loss, sdaa_compare_loss)

    print("MeanRelativeError:", mean_relative_error)
    print("MeanAbsoluteError:", mean_absolute_error)

    if mean_relative_error <= mean_absolute_error:
        print("Rule,mean_relative_error", mean_relative_error)
    else:
        print("Rule,mean_absolute_error", mean_absolute_error)

    print_str = f"{mean_relative_error=} <= 0.05 or {mean_absolute_error=} <= 0.0002"
    if mean_relative_error <= 0.05 or mean_absolute_error <= 0.0002:
        print('pass', print_str)
        return True, print_str
    else:
        print('fail', print_str)
        return False, print_str

def parse_string(string):
    pattern = r"CriticLoss: ([+-]?\d+\.\d+)"
    matches = re.findall(pattern, string, re.IGNORECASE)
    return matches

def parse_loss(ret_list):
    loss_arr = np.array([float(loss.strip()) for loss in ret_list], dtype=np.float32)
    print("Parsed loss array (first 10):", loss_arr[:10])
    return loss_arr


def plot_loss(sdaa_loss,a100_loss):
    fig, ax = plt.subplots(figsize=(12, 6))

    smoothed_losses = savgol_filter(sdaa_loss, 20, 1)
    x = list(range(len(sdaa_loss)))
    ax.plot(x, smoothed_losses, label="sdaa_loss")

    smoothed_losses = savgol_filter(a100_loss, 20, 1)
    x = list(range(len(a100_loss)))
    ax.plot(x, smoothed_losses, "--", label="cuda_loss")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Model Training Loss Curves (Smoothed)")
    ax.legend()
    plt.savefig("loss.jpg")

if __name__=="__main__":
    from argparse import ArgumentParser,ArgumentTypeError
    parser = ArgumentParser(description='modelzoo')
    parser.add_argument('--sdaa-log', type=str,default="/localnvme/application/huangyun/ICCV2019-LearningToPaint/sdaa.log")
    parser.add_argument('--cuda-log', type=str,default="/localnvme/application/huangyun/ICCV2019-LearningToPaint/cdua.log")
    args=parser.parse_args()

    sdaa_log = args.sdaa_log   
    with open(sdaa_log, 'r') as f:
        s = f.read()
    sdaa_res = parse_string(s)

    a100_log = args.cuda_log  
    with open(a100_log, 'r') as f:
        s = f.read()
    a100_res = parse_string(s)
    length=min(len(a100_res),len(sdaa_res))
    sdaa_loss = parse_loss(sdaa_res[:length])
    a100_loss = parse_loss(a100_res[:length])
    compare_loss(a100_loss, sdaa_loss) 
    plot_loss(sdaa_loss,a100_loss) 