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

import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    # 遍历10组数据文件
    for i in range(10):
        # 震源数据
        src = np.loadtxt('./eq_data/value_{num1}'.format(num1=i))
        # 台站数据
        obs = np.loadtxt('./eq_data/value_{num2}'.format(num2=i+10))
        # create database
        data_len = len(src)
  
        dataset = np.zeros((data_len, 2))
        dataset[:, 0] = obs
        dataset[:, 1] = src
        dataset = dataset.astype('float32')

        TIME_OFFSET = 1500

        for j in range(100):

            start = TIME_OFFSET * j
            test_x = dataset[start:start + 1500, 0]
            test_y = dataset[start:start + 1500, 1]

            # ----------------- plot -------------------
            plt.figure()

            plt.plot(test_x, 'k', label='obs')
            plt.plot(test_y, 'm', label='src')
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend(loc='upper right')
            plt.show()

            plt.savefig(f'./visualize_compare/compare_{j}/plot_{i}.png')

            plt.close()