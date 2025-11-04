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


class DataGenerator(object):

    def __init__(self, config):
        self.default_seed = config.get("default_seed", 42)
        self.default_float_low = config.get("default_float_low", 0.0)
        self.default_float_high = config.get("default_float_high", 1.0)
        self.default_int_low = config.get("default_int_low", 0)
        self.default_int_high = config.get("default_int_high", 100)

    def generate(self, shape, dtype, low=None, high=None, seed=None):
        if seed is None:
            seed = self.default_seed
        rng = np.random.default_rng(seed)

        if np.issubdtype(dtype, np.floating):
            if low is None:
                low = self.default_float_low
            if high is None:
                high = self.default_float_high

            return (np.array(rng.uniform(low, high, shape)).astype(dtype), seed, low, high)
        elif np.issubdtype(dtype, np.integer):
            if low is None:
                low = self.default_int_low
            if high is None:
                high = self.default_int_high

            return (np.array(rng.integers(low, high, shape, dtype, endpoint=True)), seed, low, high)
        else:
            raise ValueError("unsupport dtype {}".format(dtype))
