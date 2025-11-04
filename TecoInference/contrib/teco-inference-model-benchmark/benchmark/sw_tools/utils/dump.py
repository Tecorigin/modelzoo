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

import logging
import os
from typing import Any, Dict
import tvm
import numpy as np


def parse_data(data: Any, prefix: str, suffix: str) -> Dict[str, np.ndarray]:
    ndarray_dict = {}

    def _parse_impl(data: Any, prefix: str, suffix: str) -> None:
        nonlocal ndarray_dict
        if isinstance(data, np.ndarray):
            name = prefix + suffix
            if name in ndarray_dict:
                logging.warning("%(name) already exists in ndarray_dict.")
            ndarray_dict[name] = data
        elif isinstance(data, tvm.runtime.ndarray.NDArray):
            _parse_impl(data.numpy(), prefix, suffix)
        elif isinstance(data, dict):
            for name, val in data.items():
                new_prefix = prefix + "_" + name
                _parse_impl(val, new_prefix, suffix)
        elif isinstance(data, (tuple, list)):
            for i, val in enumerate(data):
                new_prefix = prefix + "_" + str(i)
                _parse_impl(val, new_prefix, suffix)
        else:
            raise ValueError(print("%{} type data doesn't support dumping.".format(data)))

    _parse_impl(data, prefix, suffix)

    return ndarray_dict


def dump_to_bin(config, data, save_as_txt=False):
    des_path = os.path.join(config["dump_root"], "infer_results")
    os.makedirs(des_path, exist_ok=True)
    for k, v in data.items():
        data_dict = parse_data(v, k, ".txt" if save_as_txt else ".npy")
        for tensor_name, data_val in data_dict.items():
            if save_as_txt:
                np.savetxt(
                    os.path.join(des_path, tensor_name),
                    data_val.reshape(1, -1),
                    fmt="%.8f",
                    delimiter="\n",
                )
                logging.info("save infer %(k) to %(des_path) as *.npy complete.")
            else:
                np.save(os.path.join(des_path, tensor_name), data)
                logging.info("save infer %(k) to %(des_path) as *.npy complete.")
