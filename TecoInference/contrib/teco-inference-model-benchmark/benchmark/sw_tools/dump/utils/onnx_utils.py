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


def get_dtype(proto):
    if isinstance(proto, onnx.ValueInfoProto):
        onnx_dtype = proto.type.tensor_type.elem_type
    elif isinstance(proto, onnx.TensorProto):
        onnx_dtype = proto.data_type
    else:
        onnx_dtype = None

    if onnx_dtype in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_dtype]

    return None


def get_shape(proto):
    if isinstance(proto, onnx.ValueInfoProto):
        shape = []
        for d in proto.type.tensor_type.shape.dim:  # TensorShapeProto
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape.append(d.dim_param)
            else:
                shape.append(-1)

        return shape
    elif isinstance(proto, onnx.TensorProto):
        return list(proto.dims)

    return None


def get_initializer_names(model):
    return [init.name for init in model.graph.initializer]


def get_input_names(model):
    return [i.name for i in model.graph.input]


def get_input_map(model):
    input_map = {}
    for i in model.graph.input:
        if i.name in input_map:
            raise ValueError("duplicate input {}".format(i.name))

        input_map[i.name] = {"dtype": get_dtype(i), "shape": get_shape(i)}

    return input_map


def get_node_map(model):
    node_map = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            if node.name in node_map:
                raise ValueError("duplicate node {}".format(node.name))

            node_map[node.name] = node

    return node_map


def get_node_output_map(model):
    node_output_map = {}
    for node in model.graph.node:
        for o in node.output:
            if str(o) in node_output_map:
                raise ValueError("duplicate output {}".format(str(o)))

            node_output_map[str(o)] = node

    return node_output_map


def get_node_output_names(model):
    node_output_names = []
    for node in model.graph.node:
        if node.op_type != "Constant":
            node_output_names.extend(list(node.output))

    return node_output_names


def get_node_constant_names(model):
    constant_names = []
    for node in model.graph.node:
        if node.op_type == "Constant":
            constant_names.extend(node.output)

    return constant_names


def get_output_names(model):
    return [o.name for o in model.graph.output]


def get_value_info_map(model):
    value_info_map = {}
    for i in model.graph.value_info:
        if i.name in value_info_map:
            raise ValueError("duplicate value_info {}".format(i.name))

        value_info_map[i.name] = i

    return value_info_map


def extract_model(model, input_names, output_names):
    extractor = onnx.utils.Extractor(model)

    return extractor.extract_model(input_names, output_names)


def mark_outputs(model, output_names, value_info_map=None):
    if value_info_map is None:
        value_info_map = get_value_info_map(model)

    output_value_info = []
    for output_name in output_names:
        if output_name in value_info_map:
            output_value_info.append(value_info_map[output_name])
        else:
            output_value_info.append(onnx.helper.make_empty_tensor_value_info(output_name))

    if not output_value_info:
        raise ValueError("no valid output value_info")

    while model.graph.output:
        model.graph.output.pop()

    model.graph.output.extend(output_value_info)

    return model


def serialize_to_str(model):
    return model.SerializeToString()


def deserialize_from_str(model_string):
    model = onnx.ModelProto()
    model.ParseFromString(model_string)

    return model
