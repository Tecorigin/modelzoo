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


class RandomBoundary(object):

    class Boundary:

        def __init__(self, low, high):
            self.low = low
            self.high = high

    def __init__(self,
                 default_float_low=-1.0,
                 default_float_high=1.0,
                 default_int_low=0,
                 default_int_high=256):
        self.default_float_low = default_float_low
        self.default_float_high = default_float_high
        self.default_int_low = default_int_low
        self.default_int_high = default_int_high

        self.dtype_dict = {}
        self.float_dict = {}
        self.int_dict = {}
        self.bool_dict = {}

    def add(self, index, dtype, low=None, high=None):
        if dtype == "float16" or \
           dtype == "float32" or \
           dtype == "float64":
            self.add_float(index, low, high)
        elif dtype == "int8" or \
             dtype == "int16" or \
             dtype == "int32" or \
             dtype == "int64" or \
             dtype == "uint1" or \
            dtype == "bool" :
            self.add_int(index, low, high)
        else:
            raise ValueError("unsupport dtype {}".format(dtype))
        self.dtype_dict[index] = dtype

    def add_float(self, index, low=None, high=None):
        if low is None:
            low = self.default_float_low

        if high is None:
            high = self.default_float_high

        self.float_dict[index] = self.Boundary(low, high)

    def add_int(self, index, low=None, high=None):
        if low is None:
            low = self.default_int_low

        if high is None:
            high = self.default_int_high

        self.int_dict[index] = self.Boundary(low, high)

    def get(self, index, dtype):
        ret = None

        if dtype == "float16" or \
           dtype == "float32" or \
           dtype == "float64":
            ret = self.get_float(index)
        elif dtype == "int8" or \
             dtype == "int16" or \
             dtype == "int32" or \
             dtype == "int64" or \
             dtype == "uint1" or \
             dtype == "bool" :
            ret = self.get_int(index)
        else:
            raise ValueError("unsupport dtype {}".format(dtype))

        return ret

    def get_float(self, index):
        ret = self.float_dict.get(index)

        if ret is None:
            ret = self.Boundary(self.default_float_low, self.default_float_high)

        return ret

    def get_int(self, index):
        ret = self.int_dict.get(index)

        if ret is None:
            ret = self.Boundary(self.default_int_low, self.default_int_high)

        return ret

    def get_dtype(self, index):
        ret = self.dtype_dict.get(index)

        if ret is None:
            raise ValueError("cannot get boundary dtype from name {}".format(index))

        return ret


def json_dict_to_random_boundary(json_dict):
    random_boundary = RandomBoundary()

    json_default_float = json_dict.get("default_float")
    if json_default_float is not None:
        random_boundary.default_float_low = json_default_float[0]
        random_boundary.default_float_high = json_default_float[1]

    json_default_int = json_dict.get("default_int")
    if json_default_int is not None:
        random_boundary.default_int_low = json_default_int[0]
        random_boundary.default_int_high = json_default_int[1]

    json_inputs = json_dict.get("inputs")
    if json_inputs is not None:
        for input_name, input_attr in json_inputs.items():
            dtype = input_attr.get("dtype")
            low = input_attr.get("low")
            high = input_attr.get("high")

            random_boundary.add(input_name, dtype, low, high)

    return random_boundary
