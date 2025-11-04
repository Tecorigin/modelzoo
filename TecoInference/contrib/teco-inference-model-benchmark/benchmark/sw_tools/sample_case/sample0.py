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
    dump_info = dump.DumpInfo(root_dir="model_sample_0")
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

    # 设置输入，使用DumpInfo中的随机数据信息产生输入数据
    # 1. 若DumpInfo未设置，则使用dump/utils/data.py中DataGenerator类初始化的默认值
    pipeline.set_input_data()

    # 2. 指定某个输出的随机信息
    # input_data_range = {
    #     "eager_tmp_0": {
    #         "seed": 21,
    #         "low": -1.0,
    #         "high": 0.0,
    #     }
    # }
    # pipeline.set_input_data(input_data_range=input_data_range)

    # 3. 指定某个输出的数据
    # external_input_data = {
    #    "eager_tmp_0": np.ones((1, 3, 48, 192), dtype="float32")
    # }
    # pipeline.set_input_data(external_input_data=external_input_data)

    # BasePipeline中提供的dump模式，将全部node的输出tensor依次计算结果并存储
    pipeline.dump_full()


if __name__ == "__main__":
    main()
