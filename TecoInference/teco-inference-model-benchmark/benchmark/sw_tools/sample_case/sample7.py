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
from dump.onnxruntime_executor import OnnxRuntimeExecutor


def main():
    # 模型路径
    model_name = "model_sample.onnx"

    # 加载 onnx 模型
    model = onnx.load(model_name)

    # 初始化 DumpInfo，添加模型和随机数据的相关信息
    dump_info = dump.DumpInfo(root_dir="model_sample_7")
    dump_info.set_meta_info({
        "model_name": model_name,
        "model_dtype": "float16",
        "default_seed": 42,
        "default_float_min": 0.0,
        "default_float_max": 1.0,
        "default_int_min": 0,
        "default_int_max": 100,
    })

    # 初始化 Pipeline
    pipeline = dump.BasePipeline(model, OnnxRuntimeExecutor(), dump_info)

    # 按拓扑排序获取输出tensor名称
    node_output_names = pipeline.get_node_output_names()
    print("number of node_output_names: {}".format(len(node_output_names)))
    print("node_output_names:\n {}".format(node_output_names))

    # 设置输入，使用DumpInfo中的随机数据信息产生输入数据
    pipeline.set_input_data()

    # BasePipeline中提供的dump模式，将指定范围内的node的输出tensor依次计算结果并存储
    # 1. 开始和结束通过名称设置，步长使用默认值1
    pipeline.dump_range(start_tensor="hardswish_0.tmp_0", end_tensor="hardswish_1.tmp_0")

    # 2. 开始和结束通过拓扑索引编号设置
    # pipeline.dump_range(start_tensor=2, end_tensor=20, step=5)

    # 3. 开始用名称设置，结束通过拓扑索引编号设置
    # pipeline.dump_range(start_tensor="hardswish_0.tmp_0", end_tensor=40, step=3)

    # 4. 不设置开始，默认为从第一个输出
    # pipeline.dump_range(start_tensor=None, end_tensor=40, step=2)

    # 5. 不设置结束，默认为直到最后一个输出
    # pipeline.dump_range(start_tensor="p2o.Add.7", end_tensor=None, step=2)

    # 6. 不设置开始和结束，默认全部范围
    # pipeline.dump_range(start_tensor=None, end_tensor=None, step=10)


if __name__ == "__main__":
    main()
