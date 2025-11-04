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
# pylint: disable=dangerous-default-value
import json
import time
import numpy as np

from .utils.file import join_path, make_dir


class DumpInfo(object):

    def __init__(self,
                 root_dir=None,
                 inputs_dir="inputs",
                 outputs_dir="outputs",
                 dump_info_json="dump_info.json"):
        self.root_dir = root_dir
        self.inputs_dir = join_path(inputs_dir, root_dir)
        self.outputs_dir = join_path(outputs_dir, root_dir)
        self.dump_info_json = join_path(dump_info_json, root_dir)

        self.meta_info = {
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        self.input_info = []
        self.output_info = []
        self.node_output_info = []

        make_dir(self.root_dir, verbose=True)
        make_dir(self.inputs_dir, verbose=True)
        make_dir(self.outputs_dir, verbose=True)

    def set_meta_info(self, meta_info={}):

        def _set_meta_info(key):
            if key in meta_info:
                self.meta_info[key] = meta_info.get(key)

        _set_meta_info("model_name")
        _set_meta_info("model_dtype")
        _set_meta_info("default_seed")
        _set_meta_info("default_float_min")
        _set_meta_info("default_float_max")
        _set_meta_info("default_int_min")
        _set_meta_info("default_int_max")

        _set_meta_info("executor")
        _set_meta_info("extract_subgraph")
        _set_meta_info("input_names")
        _set_meta_info("output_names")
        _set_meta_info("node_output_names")

        self.dump()

    def _get_tensor_info(self, array):
        tensor_info = {}
        tensor_info["dtype"] = str(array.dtype)
        tensor_info["shape"] = list(array.shape)
        tensor_info["has_nan"] = bool(np.any(np.isnan(array)))
        tensor_info["has_inf"] = bool(np.any(np.isinf(array)))
        tensor_info["mean"] = float(np.mean(array))
        tensor_info["median"] = float(np.median(array))
        tensor_info["min"] = float(np.min(array))
        tensor_info["max"] = float(np.max(array))

        return tensor_info

    def add_input_info(self, input_name, input_array, input_config={}):
        self.input_info.append({
            "input_name": input_name,
            "seed": input_config.get("seed"),
            "low": input_config.get("low"),
            "high": input_config.get("high"),
            "use_external": input_config.get("use_external", False),
            "tensor_info": self._get_tensor_info(input_array),
        })

        np.save(join_path(input_name, self.inputs_dir), input_array)

        self.dump()

    def clear_input_info(self):
        self.input_info.clear()

    def add_node_output_info(self,
                             node_output_name,
                             node_proto,
                             node_output_array,
                             topo_index=None):
        output_info = {}
        output_info["node_output_name"] = node_output_name
        if node_proto is not None:
            output_info["node_name"] = str(node_proto.name)
            output_info["node_type"] = str(node_proto.op_type)
            output_info["node_input_name"] = list(node_proto.input)
        output_info["tensor_info"] = self._get_tensor_info(node_output_array)

        self.node_output_info.append(output_info)

        if topo_index is None:
            file_name = node_output_name
        else:
            file_name = "topoidx_{:0>5d}__{}".format(topo_index, node_output_name)

        np.save(join_path(file_name, self.outputs_dir), node_output_array)

        self.dump()

    def add_node_output_error(self, node_output_name, err=None):
        self.node_output_info.append({
            "node_output_name": node_output_name,
            "error": str(err),
        })

        self.dump()

    def clear_node_output_info(self):
        self.node_output_info.clear()

    def dump(self):
        make_dir(self.root_dir)
        make_dir(self.inputs_dir)
        make_dir(self.outputs_dir)

        dump_info = {
            "meta_info": self.meta_info,
            "input_info": self.input_info,
            "node_output_info": self.node_output_info,
        }

        with open(self.dump_info_json, "w") as f:
            json.dump(dump_info, f, indent=4)
