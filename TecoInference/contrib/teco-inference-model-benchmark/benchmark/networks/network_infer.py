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
# pylint: disable=unused-import
import logging
import onnx
import tvm
from tvm import relay
from tvm.relay.transform import InferType
from networks.create_network import BaseIRmodule, build_engine, build_engine_dyn
from utils import load_module_from_file, \
                  topological_sorting, \
                  exist_engine, \
                  save_engine, \
                  load_engine, \
                  get_input_np_dtype_from_onnx, \
                  get_input_np_shape_from_onnx, \
                  gen_network_inputs_from_given_shape, \
                  exist_engine_fbs, \
                  save_engine_fbs, \
                  load_engine_fbs
import numpy as np


class Network(BaseIRmodule):

    def pass_process(self):
        if self.toposort:
            topological_sorting(self.onnx_model)
        passes_module = load_module_from_file(self.pass_path)
        if hasattr(passes_module, 'specified_op_dtype_dict'):
            specified_op_dtype_dict=passes_module.specified_op_dtype_dict
        else:
            specified_op_dtype_dict={}
        ir_module, param = relay.frontend.from_onnx(
            self.onnx_model, 
            self.input_shape, 
            dtype=self.dtype, 
            specified_op_dtype=specified_op_dtype_dict
            )
        ir_module = passes_module.use_passes(ir_module)
        if hasattr(passes_module,'convert_judge'):
            convert_judge=passes_module.convert_judge
        else:
            convert_judge=None
        if hasattr(passes_module,'skip_ops'):
            skip_ops=passes_module.skip_ops
        else:
            skip_ops=None
        if hasattr(passes_module,'convert_all'):
            convert_all=passes_module.convert_all
        else:
            convert_all=False
        if hasattr(passes_module, 'need_convert'):
            need_convert = passes_module.need_convert
        else:
            need_convert = True
        if hasattr(passes_module, 'normalized'):
            normalized=passes_module.normalized
        else:
            normalized=True
        return ir_module, param, convert_judge, skip_ops, convert_all, need_convert, normalized

    def ir_module_build_dyn(self, sdaa_target, sdaa_device_type, sdaa_disabled_pass):
        ir_module, param, convert_judge, pass_skip_ops, convert_all, need_convert, normalized = self.pass_process()
        if need_convert and ir_module is not None and hasattr(self, 'dtype') and self.dtype=='float16':
            ir_module = relay.frontend.convert_float_to_float16(ir_module,skip_ops=pass_skip_ops,convert_all=convert_all,convert_judge=convert_judge,normalized=normalized)
        ir_module = InferType()(ir_module)

        for param_iter in ir_module["main"].params:
            shape = [int(x) for x in param_iter.checked_type.shape]
            dtype = param_iter.checked_type.dtype
            name = param_iter.name_hint
            logging.debug(f"ir_module input name {name}, shape {shape}, dtype {dtype}")
        engine, model_fbs = build_engine_dyn(ir_module,
                                param,
                                target=sdaa_target,
                                device_type=sdaa_device_type,
                                disabled_pass=sdaa_disabled_pass,
                                engine_config=self.test_configs)

        if self.use_cache or self.test_configs.get('save_engine_path'):
            save_engine_path = self.test_configs.get('save_engine_path') or self.model_path
            save_engine_fbs(model_fbs, save_engine_path, self.input_shape)

        return engine

    def infer_dyn(self, case_name):
        self.onnx_model = onnx.load(self.model_path)
        sdaa_target = tvm.target.Target("sdaa --libs=tecodnn,tecoblas", host="llvm")
        sdaa_device_type = "sdaa"
        sdaa_disabled_pass = ["SimplifyInference"]

        if self.use_device_id is None:
            sdaa_device = tvm.device(sdaa_device_type)
        else:
            sdaa_device = tvm.device(sdaa_device_type, self.use_device_id)

        if self.use_cache and exist_engine_fbs(self.test_configs.get('load_engine_path') or self.model_path, self.input_shape):
            sdaa_engine = load_engine_fbs(self.test_configs.get('load_engine_path') or self.model_path, self.input_shape, self.test_configs)
        else:
            sdaa_engine = self.ir_module_build_dyn(sdaa_target=sdaa_target,
                                                   sdaa_device_type=sdaa_device_type,
                                                   sdaa_disabled_pass=sdaa_disabled_pass)

        input_dtype = get_input_np_dtype_from_onnx(self.onnx_model)
        for name, dt in input_dtype.items():
            if hasattr(self, 'dtype') and self.dtype=='float16' and dt=='float32':
                input_dtype[name] = 'float16'
            if self.boundary is not None and self.boundary.get_dtype(name)=='float32':
                input_dtype[name] = 'float32'

        input_shape = self.input_shape if self.input_shape else get_input_np_shape_from_onnx(self.onnx_model)
        sdaa_inputs = gen_network_inputs_from_given_shape(input_shape, input_dtype, self.boundary)
        if self.use_async:
            import os
            import collections
            self.max_engine_nums = 4
            max_engine_nums = os.getenv("MAX_ENGINE_NUMS")
            if max_engine_nums is None:
                logging.debug("MAX_ENGINE_NUMS not exist")
            elif max_engine_nums.isdigit():
                self.max_engine_nums = int(max_engine_nums)
                if self.max_engine_nums <= 0:
                    logging.error(f"illegal MAX_ENGINE_NUMS: {self.max_engine_nums}")
            else:
                logging.warning(f"illegal MAX_ENGINE_NUMS: {self.max_engine_nums}")
            logging.info(f"MAX_ENGINE_NUMS: {self.max_engine_nums}")
            
            gen_inputs = collections.OrderedDict()
            for i in range(self.max_engine_nums):
                for name, data in sdaa_inputs.items():
                    if name in gen_inputs:
                        gen_inputs[name] = np.concatenate([gen_inputs[name], data], axis=0)
                    else:
                        gen_inputs[name] = data
            sdaa_inputs.clear()
            sdaa_inputs = gen_inputs

        if self.extra_params:
            if 'inputs' in self.extra_params.keys():
                for k, v in self.extra_params['inputs'].items():
                    logging.debug(f"Load input name {k}, path {v}")
                    if v.endswith('.npy'):
                        sdaa_inputs[k] = np.load(v)

        sdaa_outputs, h2d_time, infer_time, d2h_time = self.run_dyn(sdaa_engine=sdaa_engine,
                                                                    inputs=sdaa_inputs,
                                                                    device=sdaa_device,
                                                                    case_name=case_name)

        ort_outputs, ort_time = self.onnx_infer(sdaa_inputs)

        for iter_ in range(len(sdaa_outputs)):
            self.assert_output(sdaa_outputs[iter_].astype("float32"),
                               ort_outputs[iter_].astype("float32"))
        logging.info(f"TecoInfer H2D time is: {h2d_time} ms")
        logging.info(f"TecoInfer INFER time is: {infer_time} ms")
        logging.info(f"TecoInfer D2H time is: {d2h_time} ms")
        logging.info(f"TecoInfer E2E time is: {h2d_time + infer_time + d2h_time} ms")
        logging.info(f"ORT E2E time is: {ort_time} ms")

    def ir_module_build(self, sdaa_target, sdaa_device_type, sdaa_disabled_pass):
        ir_module, param, convert_judge, pass_skip_ops, convert_all, need_convert, normalized = self.pass_process()
        if need_convert and ir_module is not None and hasattr(self, 'dtype') and self.dtype=='float16':
            ir_module = relay.frontend.convert_float_to_float16(ir_module,skip_ops=pass_skip_ops,convert_all=convert_all,convert_judge=convert_judge,normalized=normalized)
        ir_module = InferType()(ir_module)

        sdaa_engine, lib = build_engine(ir_module,
                                param,
                                target=sdaa_target,
                                device_type=sdaa_device_type,
                                disabled_pass=sdaa_disabled_pass)

        if self.use_cache or self.test_configs.get('save_engine_path'):
            save_engine_path = self.test_configs.get('save_engine_path') or self.model_path
            save_engine(lib, save_engine_path)

        for param_iter in ir_module["main"].params:
            shape = [int(x) for x in param_iter.checked_type.shape]
            dtype = param_iter.checked_type.dtype
            name = param_iter.name_hint
            logging.debug(f"ir_module input name {name}, shape {shape}, dtype {dtype}")
        return sdaa_engine

    def infer(self, case_name):
        if self.use_dyn is True:
            self.infer_dyn(case_name)
            return

        self.onnx_model = onnx.load(self.model_path)
        sdaa_target = tvm.target.Target("sdaa --libs=tecodnn,tecoblas", host="llvm")
        sdaa_device_type = "sdaa"
        sdaa_disabled_pass = ["SimplifyInference"]

        if self.use_device_id is None:
            sdaa_device = tvm.device(sdaa_device_type)
        else:
            sdaa_device = tvm.device(sdaa_device_type, self.use_device_id)

        if self.use_cache and exist_engine(self.test_configs.get('load_engine_path') or self.model_path):
            self.sdaa_engine = load_engine(self.test_configs.get('load_engine_path') or self.model_path)
        else:
            self.sdaa_engine = self.ir_module_build(sdaa_target=sdaa_target,
                                           sdaa_device_type=sdaa_device_type,
                                           sdaa_disabled_pass=sdaa_disabled_pass)
        sdaa_inputs = self.gen_network_inputs(self.sdaa_engine, self.boundary, onnx_model=self.onnx_model)
        input={}
        if self.extra_params:
            if 'inputs' in self.extra_params.keys():
                for k, v in self.extra_params['inputs'].items():
                    logging.debug(f"Load input name {k}, path {v}")
                    if v.endswith('.npy'):
                        sdaa_inputs[k] = np.load(v)

        sdaa_outputs, h2d_time, infer_time, d2h_time = self.tvm_infer(sdaa_engine=self.sdaa_engine,
                                                                    inputs=sdaa_inputs,
                                                                    device=sdaa_device,
                                                                    case_name=case_name)
        ort_outputs, ort_time = self.onnx_infer(sdaa_inputs)
        for iter_ in range(len(sdaa_outputs)):
            self.assert_output(sdaa_outputs[iter_].astype("float32"),
                            ort_outputs[iter_].astype("float32"))
        logging.info(f"TVM H2D time is: {h2d_time} ms")
        logging.info(f"TVM INFER time is: {infer_time} ms")
        logging.info(f"TVM D2H time is: {d2h_time} ms")
        logging.info(f"TVM E2E time is: {h2d_time + infer_time + d2h_time} ms")
        logging.info(f"ORT E2E time is: {ort_time} ms")
