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
import os
import matplotlib.pyplot as plt

data = []
for i in range(10):

    dir = f'./data/value_{i}'
    eq = np.loadtxt(dir)
    
    value = []
    for j in range(100):
        value.append(eq[j * 1500 : j * 1500 + 1500])

    data.append(value)

# shape = [10, 100, 1500]
eq_dataset = np.array(data)
# shape = [100, 1500, 10]
eq_dataset = eq_dataset.transpose(1, 0, 2)

'''==========================================================='''

data = []
for i in range(10):

    dir = f'./data/value_{i+10}'
    ob = np.loadtxt(dir)
    
    value = []
    for j in range(100):
        value.append(ob[j * 1500 : j * 1500 + 1500])

    data.append(value)

# shape = [10, 100, 1500]
ob_dataset = np.array(data)
# shape = [100, 1500, 10]
ob_dataset = ob_dataset.transpose(1, 0, 2)


plt.figure()

plt.plot(eq_dataset[0][0], color='b', label='eq_data')
plt.plot(ob_dataset[0][0], color='r', label='pre_test')
plt.xlabel('time')
plt.ylabel('value')
plt.legend(loc='upper right')
plt.show()
plt.savefig('./result_visualize_transf/plot_0.png')

plt.close()



