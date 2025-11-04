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
import paddle
'''
use from https://github.com/autumn-DL/TorchInterp
it use MIT license
'''

def select_scatter(inputs:paddle.Tensor, cond:paddle.Tensor, value:paddle.Tensor):
    idx = paddle.where(cond)[0]
    gather_val = paddle.gather_nd(inputs, idx)
    gather_val_new = value - gather_val
    out = paddle.scatter_nd_add(inputs, idx, gather_val_new)
    return out

def paddle_interp(x, xp, fp):
    # if not isinstance(x, torch.Tensor):
    #     x = torch.tensor(x)
    # if not isinstance(xp, torch.Tensor):
    #     xp = torch.tensor(xp)
    # if not isinstance(fp, torch.Tensor):
    #     fp = torch.tensor(fp)

    sort_idx = paddle.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[sort_idx]
    right_idxs = paddle.searchsorted(xp, x)
    right_idxs = right_idxs.clip(max=xp.shape[0] - 1)

    left_idxs = (right_idxs - 1).clip(min=0)

    x_left = xp[left_idxs]
    x_right = xp[right_idxs]
    y_left = fp[left_idxs]
    y_right = fp[right_idxs]

    interp_vals = y_left + ((x - x_left) * (y_right - y_left) / (x_right - x_left))
    # interp_vals[x < xp[0]] = fp[0]
    interp_vals = paddle.where(x < xp[0], fp[0], interp_vals)
    # interp_vals[x > xp[-1]] = fp[-1]
    interp_vals = paddle.where(x > xp[-1], fp[-1], interp_vals)
    return interp_vals


def batch_interp_with_replacement_detach(uv, f0):
    '''
    :param uv: B T
    :param f0: B T
    :return: f0 B T
    '''

    result = f0.clone() # paddle的赋值有bug，赋值之后和没有赋值一个效果
    for i in range(uv.shape[0]):
        x = paddle.where(uv[i])[-1]
        xp = paddle.where(paddle.logical_not(uv[i]))[-1]
        fp = paddle.masked_select(f0[i],paddle.logical_not(uv[i]))
        x = x if x.dim() == 1 else x.reshape([-1])
        xp = xp if xp.dim() == 1 else xp.reshape([-1])
        interp_vals = paddle_interp(x, xp, fp)
        # result[i][uv[i]] = interp_vals # paddle的赋值有bug，赋值之后和没有赋值一个效果
        result[i] = select_scatter(result[i],uv[i],interp_vals)

    return result


def unit_text():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
                exit(1)

    # f0
    f0 = paddle.to_tensor([1, 0, 3, 0, 0, 3, 4, 5, 0, 0]).astype(paddle.float32)
    uv = paddle.to_tensor([0, 1, 0, 1, 1, 0, 0, 0, 1, 1]).astype(paddle.bool)

    interp_f0 = batch_interp_with_replacement_detach(uv.unsqueeze(0), f0.unsqueeze(0)).squeeze(0)

    

if __name__ == '__main__':
    unit_text()
