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

import argparse
import os
import copy
import onnx
import dump
import numpy as np
from dump.onnxruntime_executor import OnnxRuntimeExecutor
from dump.sdaa_tvm_executor import SdaaTvmExecutor
from dump.compare_tools import CompareDiff3Handle


def topological_sorting(onnx_model):
    old_nodes = copy.deepcopy(onnx_model.graph.node)
    new_nodes = copy.deepcopy(old_nodes)
    while len(new_nodes) > 0:
        del new_nodes[0]

    # topological sorting
    input_names = set()
    for _input in onnx_model.graph.input:
        input_names.update([_input.name])

    for _ in range(len(old_nodes)):
        for old_node in old_nodes:
            flag = True

            for input_name in old_node.input:
                if not input_name in input_names:
                    flag = False
                    break

            # all node inputs are available
            if flag:
                new_nodes.append(old_node)
                input_names.update(old_node.output)

        for node in new_nodes:
            if node in old_nodes:
                old_nodes.remove(node)

        if len(old_nodes) == 0:
            break

    # use new_nodes instead of old_nodes
    while len(onnx_model.graph.node) > 0:
        del onnx_model.graph.node[0]
    for node in new_nodes:
        onnx_model.graph.node.append(node)

    return onnx_model


def create_parser():
    parser = argparse.ArgumentParser(description="layers's output compare and convert tools.")
    subparsers = parser.add_subparsers(help='commands')

    dump_full = subparsers.add_parser(name='dump_full', help='Layer-by-layer dump op information.')
    dump_full.add_argument('-f',
                           '--frame_type',
                           type=str,
                           required=True,
                           help="onnxruntime type is ort, sdaa type is sdaa")
    dump_full.add_argument('-d', '--onnx_dir', type=str, required=True, help="onnx model path.")
    dump_full.add_argument('-mt',
                           '--model_type',
                           type=str,
                           default="float16",
                           help="onnx model type.")
    dump_full.add_argument('-s', '--step', type=int, default=1, help="step dump model.")
    dump_full.add_argument(
        '-i',
        '--input_file',
        default="",
        help="input message. -i key:value : key=input_name, value=input_data.npy")
    dump_full.add_argument('-p',
                           '--passed_path',
                           type=str,
                           default="",
                           help="Use only in sdaa frame type, pass.py files used by sdaa.")
    dump_full.add_argument('-cd',
                           '--compare_dir',
                           default="",
                           help="onnxruntime fp16 output tensor data path.")
    dump_full.add_argument('-g',
                           '--golden_dir',
                           default="",
                           help="onnxruntime fp32 output tensor data path.")
    dump_full.add_argument('-t',
                           '--input_tag',
                           default=10.0,
                           type=float,
                           help="A multiple of sdaa and golden")
    dump_full.add_argument('-t1', '--th1', default=1e-6, type=float, help="Accuracy threshold.")
    dump_full.add_argument('-t2', '--th2', default=1e-3, type=float, help="diff2 threshold.")
    dump_full.set_defaults(name="dump_full")

    dump_once = subparsers.add_parser(name='dump_once', help='dump specifies the layer information')
    dump_once.add_argument('-f',
                           '--frame_type',
                           type=str,
                           required=True,
                           help="onnxruntime type is ort, sdaa type is sdaa")
    dump_once.add_argument('-d', '--onnx_dir', type=str, required=True, help="onnx model path.")
    dump_once.add_argument('-mt',
                           '--model_type',
                           type=str,
                           default="float16",
                           help="onnx model type.")
    dump_once.add_argument('-n',
                           '--output_name',
                           type=str,
                           required=True,
                           help="The output name of the OP that needs to be dumped.")
    dump_once.add_argument(
        '-i',
        '--input_file',
        default="",
        help="input message. -i key:value : key=input_name, value=input_data.npy")
    dump_once.add_argument('-p',
                           '--passed_path',
                           type=str,
                           default="",
                           help="Use only in sdaa frame type, pass.py files used by sdaa.")
    dump_once.add_argument('-cd',
                           '--compare_dir',
                           default="",
                           help="onnxruntime fp16 output tensor data path.")
    dump_once.add_argument('-g',
                           '--golden_dir',
                           default="",
                           help="onnxruntime fp32 output tensor data path.")
    dump_once.add_argument('-t',
                           '--input_tag',
                           default=10.0,
                           type=float,
                           help="A multiple of sdaa and golden")
    dump_once.add_argument('-t1', '--th1', default=1e-6, type=float, help="Accuracy threshold.")
    dump_once.add_argument('-t2', '--th2', default=1e-3, type=float, help="diff2 threshold.")
    dump_once.set_defaults(name="dump_once")

    dump_range = subparsers.add_parser(name='dump_range',
                                       help='dump specifies the layer information')
    dump_range.add_argument('-f',
                            '--frame_type',
                            type=str,
                            required=True,
                            help="onnxruntime type is ort, sdaa type is sdaa")
    dump_range.add_argument('-d', '--onnx_dir', type=str, required=True, help="onnx model path.")
    dump_range.add_argument('-mt',
                            '--model_type',
                            type=str,
                            default="float16",
                            help="onnx model type, default is float16")
    dump_range.add_argument('-b', '--begin', type=int, default=0, help="dump model begin")
    dump_range.add_argument('-e', '--end', type=int, default=0, help="dump model end")
    dump_range.add_argument('-bs', '--begin_string', type=str, default="", help="dump model begin")
    dump_range.add_argument('-es', '--end_string', type=str, default="", help="dump model end")
    dump_range.add_argument('-s', '--step', type=int, default=1, help="step dump model.")
    dump_range.add_argument(
        '-i',
        '--input_file',
        default="",
        help="input message. -i key:value : key=input_name, value=input_data.npy")
    dump_range.add_argument('-p',
                            '--passed_path',
                            type=str,
                            default="",
                            help="Use only in sdaa frame type, pass.py files used by sdaa.")
    dump_range.add_argument('-cd',
                            '--compare_dir',
                            default="",
                            help="onnxruntime fp16 output tensor data path.")
    dump_range.add_argument('-g',
                            '--golden_dir',
                            default="",
                            help="onnxruntime fp32 output tensor data path.")
    dump_range.add_argument('-t',
                            '--input_tag',
                            default=10.0,
                            type=float,
                            help="A multiple of sdaa and golden")
    dump_range.add_argument('-t1', '--th1', default=1e-6, type=float, help="Accuracy threshold.")
    dump_range.add_argument('-t2', '--th2', default=1e-3, type=float, help="diff2 threshold.")
    dump_range.set_defaults(name="dump_range")
    return parser


def check_input_val(input_val):
    if isinstance(input_val, str):
        if input_val != "":
            return input_val
        else:
            print()
            return None
    elif isinstance(input_val, int):
        if input_val != 0:
            return input_val
        else:
            return -1
    else:
        raise ValueError(print("Please input correct data type, like string/int"))


def process_input_file(input_file):
    if input_file == "":
        return ""
    ret_dict = {}
    split_data1 = input_file.split(",")
    for part in split_data1:
        split_data2 = part.split(":")
        if None is not isinstance(split_data2, set):
            if len(split_data2) != 2:
                raise ValueError(print("The input is wrong, suffix = {}".format(split_data2)))
            new_dic = {split_data2[0]: split_data2[1]}
            ret_dict.update(new_dic)
    return ret_dict


class OperationHandle(object):

    def __init__(self, args):
        self.args = args

    def operation(self):
        raise NotImplementedError


class BuildOnnxRuntimeInfer():

    def __init__(self, input_file, model_path, model_type, dump_info_path):
        # 加载 onnx 模型
        model = onnx.load(model_path)
        model = topological_sorting(model)
        dump_info = dump.DumpInfo(root_dir=dump_info_path)
        dump_info.set_meta_info({
            "model_name": model_path,
            "model_dtype": model_type,
            "default_seed": 42,
            "default_float_min": 0.0,
            "default_float_max": 1.0,
            "default_int_min": 0,
            "default_int_max": 100,
        })

        self.pipeline = dump.BasePipeline(model, OnnxRuntimeExecutor(), dump_info)
        # 设置输入，使用DumpInfo中的随机数据信息产生输入数据
        # 1. 若DumpInfo未设置，则使用dump/utils/data.py中DataGenerator类初始化的默认值
        if input_file == "":
            self.pipeline.set_input_data()
        else:
            external_input_data = {}
            dict_list = process_input_file(input_file)
            if None is not dict_list:
                for key, value in dict_list.items():
                    if os.path.exists(value):
                        np_val = np.load(value)
                        input_file_val = {key: np_val}
                        external_input_data.update(input_file_val)
                    else:
                        raise ValueError(
                            print("The path of input is wrong, suffix = {}".format(value)))
            self.pipeline.set_input_data(external_input_data=external_input_data)

    def getpipeline(self):
        return self.pipeline


class BuildSdaaInfer():

    def __init__(self,
                 input_file,
                 model_path,
                 model_type,
                 passes_path,
                 dump_info_path,
                 print_ir=False):
        # 加载 onnx 模型
        model = onnx.load(model_path)
        model = topological_sorting(model)
        dump_info = dump.DumpInfo(root_dir=dump_info_path)
        dump_info.set_meta_info({
            "model_name": model_path,
            "model_dtype": model_type,
            "default_seed": 42,
            "default_float_min": 0.0,
            "default_float_max": 1.0,
            "default_int_min": 0,
            "default_int_max": 100,
        })
        # sdaa 相关设置
        load_model_configs = {}
        load_model_configs["dtype"] = model_type
        load_model_configs["target"] = "sdaa --libs=tecodnn,tecoblas"
        load_model_configs["device_type"] = "sdaa"
        if passes_path == "":
            load_model_configs["passes"] = "ocr_passes.py"
        else:
            load_model_configs["passes"] = passes_path
        load_model_configs["disabled_pass"] = ["SimplifyInference"]
        load_model_configs["opt_level"] = 3
        load_model_configs["build_config"] = None
        load_model_configs["print_ir"] = print_ir

        execute_configs = {}
        execute_configs["use_device_id"] = None

        self.pipeline = dump.BasePipeline(model,
                                          SdaaTvmExecutor(),
                                          dump_info,
                                          load_model_configs=load_model_configs,
                                          execute_configs=execute_configs)
        # 设置输入，使用DumpInfo中的随机数据信息产生输入数据
        if input_file == "":
            self.pipeline.set_input_data()
        else:
            dict_list = process_input_file(input_file)
            external_input_data = {}
            if None is not dict_list:
                for key, value in dict_list.items():
                    if os.path.exists(value):
                        np_val = np.load(value)
                        input_file = {key: np_val}
                        external_input_data.update(input_file)
            self.pipeline.set_input_data(external_input_data=external_input_data)

    def getpipeline(self):
        return self.pipeline


class ProcessFullDump():

    def __init__(self, frame_type, model_type, onnx_dir, step, input_file, passes_path,
                 dump_info_path):
        self.step = step
        if frame_type == "sdaa":
            self.ort_pipline = BuildSdaaInfer(input_file, onnx_dir, model_type, passes_path,
                                              dump_info_path)
        elif frame_type == "ort":
            self.ort_pipline = BuildOnnxRuntimeInfer(input_file, onnx_dir, model_type,
                                                     dump_info_path)
        else:
            raise ValueError(
                print(
                    "Please input correct frame type, like sdaa/ort, now is {}".format(frame_type)))

    def process_data(self):
        if self.step == 1:
            self.ort_pipline.getpipeline().dump_full()
        else:
            self.ort_pipline.getpipeline().dump_range(start_tensor=None,
                                                      end_tensor=None,
                                                      step=self.step)


class DumpFullHandle(OperationHandle):

    def __init__(self, args):
        super(DumpFullHandle, self).__init__(args)
        self.frame_type = self.args.frame_type
        self.model_type = self.args.model_type
        self.onnx_dir = self.args.onnx_dir
        self.input_file = self.args.input_file
        self.step = self.args.step
        self.passed_path = self.args.passed_path
        self.compare_dir = self.args.compare_dir
        self.golden_dir = self.args.golden_dir
        self.input_tag = self.args.input_tag
        self.th1 = self.args.th1
        self.th2 = self.args.th2
        if (self.frame_type == "ort"):
            self.dump_info_path = "ort_dump"
        else:
            self.dump_info_path = "sdaa_dump"
        self.framework_process = ProcessFullDump(self.frame_type, self.model_type, self.onnx_dir,
                                                 self.step, self.input_file, self.passed_path,
                                                 self.dump_info_path)

    def operation(self):
        self.framework_process.process_data()
        if self.compare_dir != "":
            sdaa_path = self.dump_info_path + "/outputs"
            output_dir = "compare_result.log"
            diff3_handle = CompareDiff3Handle(sdaa_path, self.compare_dir, self.golden_dir,
                                              output_dir, self.input_tag, self.th1, self.th2)
            diff3_handle.operation()
            print("After the dump information is complete, check the corresponding data !")


class ProcessOnceDump():

    def __init__(self, frame_type, model_type, onnx_dir, input_file, passes_path, dump_name,
                 dump_info_path):
        if frame_type == "sdaa":
            self.ort_pipline = BuildSdaaInfer(input_file, onnx_dir, model_type, passes_path,
                                              dump_info_path, True)
        elif frame_type == "ort":
            self.ort_pipline = BuildOnnxRuntimeInfer(input_file, onnx_dir, model_type,
                                                     dump_info_path)
        else:
            raise ValueError(
                print(
                    "Please input correct frame type, like sdaa/ort, now is {}".format(frame_type)))
        self.ort_pipline.getpipeline().mark_outputs(output_names=[dump_name])

    def process_data(self):
        self.ort_pipline.getpipeline().dump_once()


class DumpOnceHandle(OperationHandle):

    def __init__(self, args):
        super(DumpOnceHandle, self).__init__(args)
        self.frame_type = self.args.frame_type
        self.model_type = self.args.model_type
        self.onnx_dir = self.args.onnx_dir
        self.input_file = self.args.input_file
        self.passed_path = self.args.passed_path
        self.output_name = self.args.output_name
        self.compare_dir = self.args.compare_dir
        self.golden_dir = self.args.golden_dir
        self.input_tag = self.args.input_tag
        self.th1 = self.args.th1
        self.th2 = self.args.th2
        if (self.frame_type == "ort"):
            self.dump_info_path = "ort_dump"
        else:
            self.dump_info_path = "sdaa_dump"
        self.framework_process = ProcessOnceDump(self.frame_type, self.model_type, self.onnx_dir,
                                                 self.input_file, self.passed_path,
                                                 self.output_name, self.dump_info_path)

    def operation(self):
        self.framework_process.process_data()
        if self.compare_dir != "":
            sdaa_path = self.dump_info_path + "/outputs"
            output_dir = "compare_result.log"
            diff3_handle = CompareDiff3Handle(sdaa_path, self.compare_dir, self.golden_dir,
                                              output_dir, self.input_tag, self.th1, self.th2)
            diff3_handle.operation()
            print("After the dump information is complete, check the corresponding data !")


class ProcessRangeDump():

    def __init__(self, frame_type, model_type, onnx_dir, input_file, begin, end, begin_str, end_str,
                 step, passes_path, dump_info_path):
        self.begin = begin
        self.end = end
        self.begin_str = begin_str
        self.end_str = end_str
        self.step = step
        self.input_file = input_file

        if frame_type == "sdaa":
            self.ort_pipline = BuildSdaaInfer(input_file, onnx_dir, model_type, passes_path,
                                              dump_info_path)
        elif frame_type == "ort":
            self.ort_pipline = BuildOnnxRuntimeInfer(input_file, onnx_dir, model_type,
                                                     dump_info_path)
        else:
            raise ValueError(
                print(
                    "Please input correct frame type, like sdaa/ort, now is {}".format(frame_type)))

    def process_data(self):
        begin_value = check_input_val(self.begin)
        end_value = check_input_val(self.end)
        if (-1 != begin_value) or (-1 != end_value):
            self.ort_pipline.getpipeline().dump_range(start_tensor=begin_value,
                                                      end_tensor=end_value,
                                                      step=self.step)
        else:
            begin_value = check_input_val(self.begin_str)
            end_value = check_input_val(self.end_str)
            self.ort_pipline.getpipeline().dump_range(start_tensor=begin_value,
                                                      end_tensor=end_value,
                                                      step=self.step)


class DumpRangeHandle(OperationHandle):

    def __init__(self, args):
        super(DumpRangeHandle, self).__init__(args)
        self.frame_type = self.args.frame_type
        self.model_type = self.args.model_type
        self.onnx_dir = self.args.onnx_dir
        self.input_file = self.args.input_file
        self.begin = self.args.begin
        self.end = self.args.end
        self.begin_str = self.args.begin_string
        self.end_str = self.args.end_string
        self.step = self.args.step
        self.passed_path = self.args.passed_path
        self.compare_dir = self.args.compare_dir
        self.golden_dir = self.args.golden_dir
        self.input_tag = self.args.input_tag
        self.th1 = self.args.th1
        self.th2 = self.args.th2
        if (self.frame_type == "ort"):
            self.dump_info_path = "ort_dump"
        else:
            self.dump_info_path = "sdaa_dump"
        self.framework_process = ProcessRangeDump(self.frame_type, self.model_type, self.onnx_dir,
                                                  self.input_file, self.begin, self.end,
                                                  self.begin_str, self.end_str, self.step,
                                                  self.passed_path, self.dump_info_path)

    def operation(self):
        self.framework_process.process_data()
        if self.compare_dir != "":
            sdaa_path = self.dump_info_path + "/outputs"
            output_dir = "compare_result.log"
            diff3_handle = CompareDiff3Handle(sdaa_path, self.compare_dir, self.golden_dir,
                                              output_dir, self.input_tag, self.th1, self.th2)
            diff3_handle.operation()
            print("After the dump information is complete, check the corresponding data !")


class OperationFactory(object):

    def process(self, args):
        if args.name == "dump_full":
            get_operation_results = DumpFullHandle(args)
        elif args.name == "dump_once":
            get_operation_results = DumpOnceHandle(args)
        elif args.name == "dump_range":
            get_operation_results = DumpRangeHandle(args)
        else:
            raise ValueError(
                print("Please input correct name, like compare/convert/single, now is {}".format(
                    args.name)))
        get_operation_results.operation()


def main():
    parser = create_parser()
    args = parser.parse_args()
    print("args: ", args)
    operation_factory = OperationFactory()
    operation_factory.process(args)


if __name__ == "__main__":
    main()
