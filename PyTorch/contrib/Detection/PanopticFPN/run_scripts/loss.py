# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import re
import numpy as np
import matplotlib.pyplot as plt
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
    # 匹配 rank 后有无空格都可以，兼容 cuda.log 和 sdaa.log
    pattern = r"rank\s*:\s*0\s+train\.loss\s*:\s*([\d\.e-]+)"
    matches = []
    for line in string.splitlines():
        match = re.search(pattern, line)
        if match:
            matches.append(match.group(1))
    print("xxxxx", matches)
    return matches

def parse_loss(ret_list):
    step_num=len(ret_list)
    loss_arr = np.zeros(shape=(step_num, ))
    i=0
    for loss in ret_list:
        loss = float(loss)
        loss_arr[i] = loss
        i+=1
        if i>=step_num: break
    print(loss_arr)
    return loss_arr


def plot_loss(sdaa_loss,a100_loss):
    fig, ax = plt.subplots(figsize=(12, 6))

    smoothed_losses = savgol_filter(sdaa_loss, 5, 1)
    x = list(range(len(sdaa_loss)))
    ax.plot(x, smoothed_losses, label="sdaa_loss")

    smoothed_losses = savgol_filter(a100_loss, 5, 1)
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
    parser.add_argument('--sdaa-log', type=str,default="sdaa.log")
    parser.add_argument('--cuda-log', type=str,default="cuda.log")
    args=parser.parse_args()

    sdaa_log = args.sdaa_log   
    with open(sdaa_log, 'r') as f:
        s = f.read()
    sdaa_res = parse_string(s)
    print("======",sdaa_res)
    a100_log = args.cuda_log  
    with open(a100_log, 'r') as f:
        s = f.read()
    a100_res = parse_string(s)
    length=min(len(a100_res),len(sdaa_res))
    sdaa_loss = parse_loss(sdaa_res[:length])

    a100_loss = parse_loss(a100_res[:length])
    compare_loss(a100_loss, sdaa_loss) # 比较loss
    plot_loss(sdaa_loss,a100_loss) # 对比loss曲线图