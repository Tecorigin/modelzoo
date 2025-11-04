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

import os
import torch_sdaa
import numpy as np
import torch
from torch import nn
import time
import argparse
import matplotlib.pyplot as plt
import math
from model_transformer import get_model
from eq_dataset import Earthquake_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="sdaa")
    parser.add_argument("--d_model", type=int, default=64, help="feature dim of transf")
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=int, default=0.1)
    parser.add_argument("--max_length", type=int, default=1500, help="max length of postion encoding in transformer")
    args = parser.parse_args()
    os.makedirs('result_visualize_transf', exist_ok=True)
    device = args.device

    # 产生测试数据
    obs_test_data, eq_test_data = Earthquake_data().__getitem__(99)
    obs_test_data = obs_test_data.unsqueeze(0).to(device)
    eq_test_data = eq_test_data.transpose(0, 1)  # ground truth

    # 模型
    model = get_model(args)
    model.load_state_dict(torch.load('./model_transf/model.pkl', map_location='cpu'))
    model = model.eval().to(device)

    '''==========================test data=========================='''
    pred_test = model(obs_test_data)

    '''===========================plot=============================='''
    pred_test = pred_test.squeeze(0).transpose(0, 1).cpu().data.numpy()

    for i in range(10):
        plt.figure()

        plt.plot(eq_test_data[i], color='b', label='eq_data')
        plt.plot(pred_test[i], color='r', label='pre_test')
        plt.xlabel('time')
        plt.ylabel('value')
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig(f'./result_visualize_transf/plot_{i}.png')

        plt.close()