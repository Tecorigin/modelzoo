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

import dump


def main():
    # 模型路径
    model_name = "model_sample.onnx"

    # 加载 onnx 模型
    model = onnx.load(model_name)

    ### onnx utils ###
    # onnx模型序列化和反序列化
    model = dump.deserialize_from_str(dump.serialize_to_str(model))

    # 模型初始化tensor名称（通常为常量）
    print("initializer names:\n{}\n".format(dump.get_initializer_names(model)))
    # 模型输入tensor名称
    print("input names:\n{}\n".format(dump.get_input_names(model)))
    # 模型输入tensor map，key为名称，value为dtype和shape
    print("input map:\n{}\n".format(dump.get_input_map(model)))
    # 节点输出tensor名称（非常量）
    print("node output names:\n{}\n".format(dump.get_node_output_names(model)))
    # 节点输出tensor名称（常量）
    print("node constant names:\n{}\n".format(dump.get_node_constant_names(model)))
    # 模型输出tensor名称（常量）
    print("output names:\n{}\n".format(dump.get_output_names(model)))
    # 模型中tensor的描述信息
    print("value info map:\n{}\n".format(dump.get_value_info_map(model)))

    # 截取模型
    model = dump.extract_model(model,
                               input_names=["eager_tmp_0"],
                               output_names=["hardswish_1.tmp_0"])
    print("output names after extract_model:\n{}\n".format(dump.get_output_names(model)))

    # 标记模型输出tensor
    model = dump.mark_outputs(model, output_names=[
        "p2o.GlobalAveragePool.1",
        "hardswish_1.tmp_0",
    ])
    print("output names after mark_outputs:\n{}\n".format(dump.get_output_names(model)))

    ### 随机数组生成 ###
    generator = dump.DataGenerator({
        "default_float_high": 3.0,
        "default_int_low": -10,
    })
    # 返回生成的array、seed、low、high
    print(generator.generate(shape=(1, 2, 3), dtype="float32", low=-1.0, seed=0))
    print(generator.generate(shape=(5, 1), dtype="int8", high=20))


if __name__ == "__main__":
    main()
