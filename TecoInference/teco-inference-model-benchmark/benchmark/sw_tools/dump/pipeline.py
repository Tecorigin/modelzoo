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
# pylint: disable=broad-exception-caught
import json
import numpy as np

from .executor.base_executor import BaseExecutor

from .utils.data import DataGenerator
from .utils.file import find_file
from .utils.onnx_utils import get_input_map, get_input_names, get_node_output_names
from .utils.onnx_utils import get_node_map, get_node_output_map, get_output_names
from .utils.onnx_utils import get_value_info_map, extract_model, mark_outputs


def SetValue(value, default):
    if value is None:
        return default

    return value


class BasePipeline(object):

    def __init__(self,
                 model,
                 executor,
                 dump_info,
                 init_configs=None,
                 load_model_configs=None,
                 execute_configs=None):
        if not isinstance(executor, BaseExecutor):
            raise ValueError("invalid executor")

        self.executor = executor
        self.dump_info = dump_info
        self.init_configs = init_configs
        self.load_model_configs = load_model_configs
        self.execute_configs = execute_configs

        self.model = model
        self.input_map = get_input_map(model)
        self.input_names = get_input_names(model)
        self.node_output_names = get_node_output_names(model)
        self.node_map = get_node_map(model)
        self.node_output_map = get_node_output_map(model)
        self.output_names = get_output_names(model)
        self.value_info_map = get_value_info_map(model)

        dump_info.set_meta_info({
            "executor": type(executor).__name__,
            "extract_subgraph": False,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "node_output_names": self.node_output_names,
        })

        # subgraph
        self.subgraph = self.model
        self.subgraph_input_map = self.input_map
        self.subgraph_input_names = self.input_names
        self.subgraph_node_output_names = self.node_output_names
        self.subgraph_node_map = self.node_map
        self.subgraph_node_output_map = self.node_output_map
        self.subgraph_output_names = self.output_names
        self.subgraph_value_info_map = self.value_info_map

        self.input_data = {}

    def extract_subgraph(self, input_names=None, output_names=None):
        input_names = SetValue(input_names, self.input_names)
        output_names = SetValue(output_names, self.output_names)

        self.subgraph = extract_model(self.model, input_names, output_names)
        self.subgraph_input_map = get_input_map(self.subgraph)
        self.subgraph_input_names = get_input_names(self.subgraph)
        self.subgraph_node_output_names = get_node_output_names(self.subgraph)
        self.subgraph_node_map = get_node_map(self.subgraph)
        self.subgraph_node_output_map = get_node_output_map(self.subgraph)
        self.subgraph_output_names = get_output_names(self.subgraph)
        self.subgraph_value_info_map = get_value_info_map(self.subgraph)

        self.dump_info.set_meta_info({
            "extract_subgraph": True,
            "input_names": self.subgraph_input_names,
            "output_names": self.subgraph_output_names,
            "node_output_names": self.subgraph_node_output_names,
        })

    def get_node_output_names(self):
        return self.subgraph_node_output_names

    def mark_outputs(self, output_names=None):
        if output_names is None:
            output_names = self.subgraph_node_output_names

        self.subgraph = mark_outputs(self.model, output_names, self.subgraph_value_info_map)
        self.subgraph_output_names = get_output_names(self.subgraph)

    def set_input_data(self, input_data_range={}, external_input_data={}):
        data_generator = DataGenerator(self.dump_info.meta_info)
        self.dump_info.clear_input_info()

        input_config = {}
        for name, info in self.subgraph_input_map.items():
            print("input \"{}\", shape {}, dtype {}".format(name, info["shape"], info["dtype"]))
            if external_input_data and name not in external_input_data:
                print("!!!!! input {}, is not in external input data," \
                    "We will use randomly generated input !!!!!".format(name))
            if name in external_input_data:
                self.input_data[name] = external_input_data.get(name)
                input_config["use_external"] = True
            elif name in input_data_range:
                (self.input_data[name],
                 input_config["seed"],
                 input_config["low"],
                 input_config["high"]) = \
                    data_generator.generate(info["shape"],
                                            info["dtype"],
                                            input_data_range[name].get("low"),
                                            input_data_range[name].get("high"),
                                            input_data_range[name].get("seed"))
            else:
                (self.input_data[name],
                 input_config["seed"],
                 input_config["low"],
                 input_config["high"]) = \
                    data_generator.generate(info["shape"],
                                            info["dtype"])

            self.dump_info.add_input_info(name, self.input_data[name], input_config)

    def set_input_data_from_json(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)

            if "input_info" not in json_data:
                raise ValueError("key \"input_info\" not in json")

            input_info = json_data.get("input_info")

        input_data_range = {}
        for info in input_info:
            name = info.get("input_name")
            use_external = info.get("use_external")
            if use_external:
                raise ValueError("input \"{}\" use external data".format(name))

            input_data_range[name] = {
                "seed": info.get("seed"),
                "low": info.get("low"),
                "high": info.get("high"),
            }

        for name in self.subgraph_input_map.keys():
            if name not in input_data_range:
                raise ValueError("\"{}\" not in json input_info".format(name))

        self.set_input_data(input_data_range=input_data_range)

    def set_input_data_from_file(self, dir_with_inputs):
        external_input_data = {}
        for name in self.subgraph_input_map.keys():
            pattern = "{}.npy".format(name)
            file = find_file(pattern, dir_with_inputs)
            if file is None:
                raise ValueError("\"{}\" not exist".format(pattern))

            external_input_data[name] = np.load(file)

        self.set_input_data(external_input_data=external_input_data)

    def dump_once(self, topo_indexes=None):
        self.executor.init(self.init_configs)
        self.executor.load_model(self.subgraph, self.load_model_configs)
        outputs = self.executor.execute(self.subgraph_output_names, self.input_data,
                                        self.execute_configs)
        self.executor.release()

        # dump
        if topo_indexes is None:
            for output_name, output in zip(self.subgraph_output_names, outputs):
                self.dump_info.add_node_output_info(output_name,
                                                    self.subgraph_node_output_map.get(output_name),
                                                    output)
                print("output {}, shape {}, dtype {}".format(output_name, output.shape,
                                                             output.dtype))
        else:
            if not isinstance(topo_indexes, list):
                topo_indexes = [topo_indexes]

            for output_name, output, topo_index in \
                zip(self.subgraph_output_names, outputs, topo_indexes):
                self.dump_info.add_node_output_info(output_name,
                                                    self.subgraph_node_output_map.get(output_name),
                                                    output, topo_index)
                print("topology index {}, output {}, shape {}, dtype {}".format(
                    topo_index, output_name, output.shape, output.dtype))

        return outputs

    def dump_full(self, dump_with_index=True):
        # dump for each node output name
        for index, node_output_name in enumerate(self.subgraph_node_output_names):
            if not dump_with_index:
                index = None

            self.mark_outputs([node_output_name])

            try:
                self.dump_once(topo_indexes=index)
            except Exception as e:
                self.dump_info.add_node_output_error(node_output_name, e)
                print("dump_once error [{}]".format(e))

    def dump_list(self, output_names=None, dump_with_index=True):
        if output_names is None:
            print("output_names is None, run dump_full")
            self.dump_full(dump_with_index)
        else:
            # dump for each node output name in list
            for index, node_output_name in enumerate(self.subgraph_node_output_names):
                if node_output_name not in output_names:
                    continue

                if not dump_with_index:
                    index = None

                self.mark_outputs([node_output_name])

                try:
                    self.dump_once(topo_indexes=index)
                except Exception as e:
                    self.dump_info.add_node_output_error(node_output_name, e)
                    print("dump_once error [{}]".format(e))

    def dump_range(self, start_tensor=None, end_tensor=None, step=1, dump_with_index=True):
        if start_tensor is None:
            start_tensor = 0

        if isinstance(start_tensor, str):
            start_tensor = self.subgraph_node_output_names.index(start_tensor)
        elif isinstance(start_tensor, int):
            assert start_tensor in range(len(self.subgraph_node_output_names))
        else:
            raise ValueError("start_tensor should be None, int or str")

        if end_tensor is None:
            end_tensor = len(self.subgraph_node_output_names) - 1

        if isinstance(end_tensor, str):
            end_tensor = self.subgraph_node_output_names.index(end_tensor)
        elif isinstance(start_tensor, int):
            assert end_tensor in range(len(self.subgraph_node_output_names))
        else:
            raise ValueError("end_tensor should be None, int or str")

        if start_tensor >= end_tensor:
            raise ValueError("start_tensor should be prior to end_tensor")

        if step <= 0:
            raise ValueError("step should greater than 0")

        # include start_tensor and end_tensor
        output_list = list(range(start_tensor, end_tensor, step))
        output_list.append(end_tensor)

        # dump for each node output name
        for index, node_output_name in enumerate(self.subgraph_node_output_names):
            if index not in output_list:
                continue

            if not dump_with_index:
                index = None

            self.mark_outputs([node_output_name])

            try:
                self.dump_once(topo_indexes=index)
            except Exception as e:
                self.dump_info.add_node_output_error(node_output_name, e)
                print("dump_once error [{}]".format(e))
