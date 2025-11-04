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

import onnx
from onnx import mapping


# topological_sorting will modify onnx_model
def topological_sorting(onnx_model: onnx.ModelProto):
    graph = onnx_model.graph
    deps_count = [0] * len(graph.node)  # dependency count of each node
    deps_to_nodes = {}  # input to node indice
    sorted_nodes = []  # initialize sorted_nodes
    for node_idx, node in enumerate(graph.node):
        # CANNOT use len(node.input) directly because input can be optional
        deps_count[node_idx] = sum(1 for _ in node.input if _)
        if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
            sorted_nodes.append(graph.node[node_idx])
            continue

        for input_name in node.input:
            if input_name not in deps_to_nodes:
                deps_to_nodes[input_name] = [node_idx]
            else:
                deps_to_nodes[input_name].append(node_idx)

    # Note: this logic only applies to top level graph
    # since a sub graph could use intializer from parent graph
    initializer_names = [init.name for init in graph.initializer]
    graph_input_names = [input_.name for input_ in graph.input]
    input_names = initializer_names + graph_input_names
    input_names.sort()
    prev_input_name = None
    for input_name in input_names:
        if prev_input_name == input_name:
            continue

        prev_input_name = input_name
        if input_name in deps_to_nodes:
            for node_idx in deps_to_nodes[input_name]:
                deps_count[node_idx] = deps_count[node_idx] - 1
                if deps_count[node_idx] == 0:
                    sorted_nodes.append(graph.node[node_idx])

    start = 0
    end = len(sorted_nodes)

    while start < end:
        for output in sorted_nodes[start].output:
            if output in deps_to_nodes:
                for node_idx in deps_to_nodes[output]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])
                        end = end + 1
        start = start + 1

    if end != len(graph.node):
        raise RuntimeError(f"""Graph is not a DAG: end={end},
            len(graph.node)={len(graph.node)},
            graph.node[end]={graph.node[end]}""")

    graph.ClearField("node")
    graph.node.extend(sorted_nodes)


def extract_model(model, input_names, output_names):
    extractor = onnx.utils.Extractor(model)
    extracted_model = extractor.extract_model(input_names, output_names)

    return extracted_model


def get_ort_input_names(ort_session):
    ort_input_names = []
    for ort_input in ort_session.get_inputs():
        print("onnxruntime input name: {}, shape: {}, type: {}".format(
            ort_input.name, ort_input.shape, ort_input.type))
        ort_input_names.append(ort_input.name)

    return ort_input_names


def get_ort_output_names(ort_session):
    ort_output_names = []
    for ort_output in ort_session.get_outputs():
        print("onnxruntime output name: {}, shape: {}, type: {}".format(
            ort_output.name, ort_output.shape, ort_output.type))
        ort_output_names.append(ort_output.name)

    return ort_output_names


def get_ort_inputs_from_sdaa(ort_input_names, sdaa_inputs, use_relay_serialize=False):
    ort_inputs = {}
    for ort_input_name in ort_input_names:
        sdaa_input_name = ort_input_name
        if use_relay_serialize:
            if len(sdaa_input_name) == 0 or not sdaa_input_name[0].isalpha():
                sdaa_input_name = "v" + sdaa_input_name

        if sdaa_input_name in sdaa_inputs:
            ort_inputs[ort_input_name] = sdaa_inputs[sdaa_input_name]
        else:
            raise ValueError("name {} not exist".format(sdaa_input_name))

    return ort_inputs


def get_input_np_dtype_from_onnx(onnx_model: onnx.ModelProto, input_name: str=None):
    if input_name == None:
        input_data_types = {}
        for input_node in onnx_model.graph.input:
            input_name = input_node.name
            input_data_type = mapping.TENSOR_TYPE_TO_NP_TYPE[input_node.type.tensor_type.elem_type]
            input_data_types[input_name] = input_data_type
        return input_data_types
    else:
        for input_node in onnx_model.graph.input:
            if input_node.name == input_name:
                return mapping.TENSOR_TYPE_TO_NP_TYPE[input_node.type.tensor_type.elem_type]

def get_input_np_shape_from_onnx(onnx_model: onnx.ModelProto, input_name: str=None):
    if input_name == None:
        input_data_shapes = {}
        for input_node in onnx_model.graph.input:
            input_name = input_node.name
            input_data_shape = []
            for dim in input_node.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    input_data_shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    raise RuntimeError(f"Dynamic onnx must set actul input shape, but got dim {dim.dim_param}")
                else:
                    raise RuntimeError(f"Dim in onnx must has field dim_value or dim_param, not support this dim {dim} with type {type(dim)}")
            input_data_shapes[input_name] = input_data_shape
        return input_data_shapes
    else:
        for input_node in onnx_model.graph.input:
            if input_node.name == input_name:
                input_data_shape = []
                for dim in input_node.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        input_data_shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        raise RuntimeError(f"Dynamic onnx must set actul input shape, but got dim {dim.dim_param}")
                    else:
                        raise RuntimeError(f"Dim in onnx must has field dim_value or dim_param, not support this dim {dim} with type {type(dim)}")
                return input_data_shape
