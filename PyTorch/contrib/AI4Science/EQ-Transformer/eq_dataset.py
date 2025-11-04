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
from torch.utils import data
import torch
import numpy as np


class Earthquake_data(data.Dataset):
    def __init__(self):
        super().__init__()

        '''====震源数据处理===='''
        data = []
        for i in range(10):
            dir = f'eq_data/value_{i}'
            eq = np.loadtxt(dir)
    
            value = []
            for j in range(100):
                value.append(eq[j * 1500 : j * 1500 + 1500])

            data.append(value)

        # shape = [10, 100, 1500]
        eq_dataset = np.array(data)
        # shape = [100, 1500, 10]
        self.eq_dataset = eq_dataset.transpose(1, 2, 0)
        '''====观测点数据处理===='''
        data = []
        for i in range(10):
            dir = f'eq_data/value_{i+10}'
            ob = np.loadtxt(dir)
            
            value = []
            for j in range(100):
                value.append(ob[j * 1500 : j * 1500 + 1500])

            data.append(value)

        # shape = [10, 100, 1500]
        ob_dataset = np.array(data)
        # shape = [100, 1500, 10]
        self.ob_dataset = ob_dataset.transpose(1, 2, 0)

        self.length = 99    # 100 - 1, 99 for train; 1 for test

    def __getitem__(self, index):
        eq_data = self.eq_dataset[index].astype('float32')
        ob_data = self.ob_dataset[index].astype('float32')

        eq_data = torch.from_numpy(eq_data)
        ob_data = torch.from_numpy(ob_data)

        # size = [1, 1500, 10]
        return ob_data, eq_data
    
    def __len__(self):
        # here, (99 * 1500) for train
        return self.length