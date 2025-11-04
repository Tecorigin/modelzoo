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

from abc import ABC, abstractmethod
import logging
import os
import time
import onnx
import numpy as np
import onnxruntime as ort
from networks.pass_path import pass_path

from utils import (
    build_engine,
    build_engine_dyn,
    run_engine,
    run_dyn_exec,
    get_ort_input_names,
    assert_rms,
    get_ort_output_names,
    get_ort_inputs_from_sdaa,
    gen_network_inputs,
    get_input_np_dtype_from_onnx,
)


class Register:

    def __init__(self, registry_name):
        self._dict = {}
        self._name = registry_name

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a registry must be a callable!\nvalue: {value}")
        if key is None:
            key = value.__name__
        if key in self._dict:
            logging.warning(f"Key {key} already in registry {self._name}")
        self._dict[key] = value

    def register(self, target):

        def add(key, value):
            self._dict[key] = value
            return value

        if callable(target):
            return add(target.__name__, target)
        return lambda x: add(target, x)

    def __getitem__(self, key):
        if key not in self._dict.keys():
            raise Exception(f"Network {key} isn't registered in {self._name}")
        return self._dict[key]


Register_func = Register("reg_func")


class BaseIRmodule(ABC):

    def __init__(self, test_configs: dict):
        self.test_configs = test_configs
        self.model_path = test_configs.get("model_path")
        self.onnx_model_path = test_configs.get("onnx_model_path")
        self.onnx_model_convert_to_fp32 = test_configs.get("onnx_model_convert_to_fp32")
        self.dtype = test_configs.get("dtype")
        self.warm_up = test_configs.get("warm_up", 0)
        self.run_loops = test_configs.get("run_loops", 1)
        self.use_device_id = test_configs.get("use_device_id")
        self.boundary = test_configs.get("boundary")
        self.pass_path = pass_path(test_configs.get("model_name"), test_configs.get("bs"))
        self.toposort = test_configs.get("toposort")
        self.ort_session = None
        self.onnx_model = None
        self.extra_params = test_configs.get("extra_params")

        self.input_shape = test_configs.get("input_shape", {})
        self.use_dyn = not (os.getenv("TECO_INFER_TVM_RUN", "").lower() in {"true", "1", "yes"})
        logging.info(f"Run with dyn: {self.use_dyn}")

        self.use_bs1_session = test_configs.get("use_bs1_session", False)
        logging.info(f"Run with bs1 Session: {self.use_bs1_session}")
        self.use_async = test_configs.get("use_async", False)

        self.use_cache = test_configs.get('use_cache', True)
        self.use_cross_memory = test_configs.get("use_cross_memory", False)
        logging.info(f"Run with async: {self.use_async}")

        self.max_engine_nums = 4
        max_engine_nums = os.getenv("MAX_ENGINE_NUMS")
        if max_engine_nums is None:
            logging.debug("MAX_ENGINE_NUMS not exist")
        elif max_engine_nums.isdigit():
            self.max_engine_nums = int(max_engine_nums)
            if self.max_engine_nums <= 0:
                logging.error(f"illegal MAX_ENGINE_NUMS: {max_engine_nums}")
        else:
            logging.warning(f"illegal MAX_ENGINE_NUMS: {max_engine_nums}")
        logging.info(f"MAX_ENGINE_NUMS: {self.max_engine_nums}")

    @abstractmethod
    def pass_process(self):
        pass

    @abstractmethod
    def infer(self, case_name):
        pass

    def gen_network_inputs(self, ir_module, boundary, onnx_model=None):
        if self.use_async:
            import collections
            # the order of entries added should be recorded
            gen_inputs = collections.OrderedDict()
            for i in range(self.max_engine_nums):
                gen_input = gen_network_inputs(ir_module, boundary, seed=42 + i, onnx_model=onnx_model, use_cross_memory=self.use_cross_memory)
                for name, data in gen_input.items():
                    if name in gen_inputs:
                        gen_inputs[name] = np.concatenate([gen_inputs[name], data], axis=0)
                    else:
                        gen_inputs[name] = data

            return gen_inputs
        else:
            return gen_network_inputs(ir_module, boundary, seed=42, onnx_model=onnx_model, use_cross_memory=self.use_cross_memory)

    def run_dyn(self, sdaa_engine, inputs, device, case_name):
        logging.debug(f"Run {case_name}...")
        sdaa_ctx = sdaa_engine.create_context()

        logging.debug("run sdaa engine...")
        sdaa_outputs, h2d_time, infer_time, d2h_time = run_dyn_exec(sdaa_ctx,
                                                                    inputs,
                                                                    device=device,
                                                                    run_loops=self.run_loops,
                                                                    warm_up=self.warm_up,
                                                                    use_async=self.use_async)
        for sdaa_output in sdaa_outputs:
            logging.debug(f"sdaa output shape: {sdaa_output.shape}, dtype: {sdaa_output.dtype}")

        logging.debug("release sdaa engine...")
        sdaa_ctx.release()
        sdaa_engine.release()
        return sdaa_outputs, h2d_time, infer_time, d2h_time

    def tvm_infer(self, sdaa_engine, inputs, device, case_name):
        logging.debug(f"Run {case_name}...")
        sdaa_ctx = sdaa_engine.create_context(enable_cross_memory=self.use_cross_memory)
        sdaa_ctx.set_context_name(case_name)
        logging.debug("run sdaa engine...")
        sdaa_outputs, h2d_time, infer_time, d2h_time = run_engine(sdaa_ctx,
                                                                  inputs,
                                                                  device=device,
                                                                  run_loops=self.run_loops,
                                                                  warm_up=self.warm_up,
                                                                  use_async=self.use_async)
        for sdaa_output in sdaa_outputs:
            logging.debug(f"sdaa output shape: {sdaa_output.shape}, dtype: {sdaa_output.dtype}")

        logging.debug("release sdaa engine...")
        sdaa_ctx.release()
        sdaa_engine.release()
        return sdaa_outputs, h2d_time, infer_time, d2h_time

    ### onnxruntime ###
    def onnx_infer(self, inputs):
        logging.debug("load onnxruntime model...")
        if self.onnx_model_path is not None:
            self.onnx_model = onnx.load(self.onnx_model_path)
        if self.ort_session is None:
            try:
                self.ort_session = ort.InferenceSession(self.onnx_model.SerializeToString())
            except:
                self.ort_session = ort.InferenceSession(self.model_path)
        ort_input_names = get_ort_input_names(self.ort_session)
        ort_output_names = get_ort_output_names(self.ort_session)

        ort_inputs = get_ort_inputs_from_sdaa(ort_input_names, inputs)
        for name, data in ort_inputs.items():
            onnx_input_dtype = get_input_np_dtype_from_onnx(self.onnx_model, name)
            ort_inputs[name] = data.astype(onnx_input_dtype)
        logging.debug("run onnxruntime model...")
        ort_start = time.time()
        if self.use_dyn:
            # use_dyn only support sync mode, gen one input for one device
            if self.use_async:
                ort_inputs_list = [dict() for _ in range(self.max_engine_nums)]
                for name, data_cat in ort_inputs.items():
                    data_list = np.split(data_cat, self.max_engine_nums, axis=0)
                    for ort_inputs, data in zip(ort_inputs_list, data_list):
                        ort_inputs[name] = data

                ort_outputs_list = [list() for _ in range(len(ort_output_names))]
                for ort_inputs in ort_inputs_list:
                    for o, output in enumerate(self.ort_session.run(ort_output_names, ort_inputs)):
                        ort_outputs_list[o].append(output)

                ort_outputs = [np.concatenate(output, axis=0) for output in ort_outputs_list]
            else:
                ort_outputs = self.ort_session.run(ort_output_names, ort_inputs)
        else:
            if self.use_async:
                ort_inputs_list = [dict() for _ in range(self.max_engine_nums)]
                for name, data_cat in ort_inputs.items():
                    data_list = np.split(data_cat, self.max_engine_nums, axis=0)
                    for ort_inputs, data in zip(ort_inputs_list, data_list):
                        ort_inputs[name] = data

                ort_outputs_list = [list() for _ in range(len(ort_output_names))]
                for ort_inputs in ort_inputs_list:
                    for o, output in enumerate(self.ort_session.run(ort_output_names, ort_inputs)):
                        ort_outputs_list[o].append(output)

                ort_outputs = [np.concatenate(output, axis=0) for output in ort_outputs_list]
            else:
                ort_outputs = self.ort_session.run(ort_output_names, ort_inputs)
        ort_time = time.time() - ort_start
        return ort_outputs, ort_time * 1000

    def assert_output(self, sdaa_outputs, ort_outputs):
        logging.debug("compare outputs...")
        threshold = 1e-4 if self.dtype == "float16" else 1e-6
        logging.info(f"case path : {self.model_path}")
        assert_rms(sdaa_outputs, ort_outputs, atol=1e-2, th=threshold)


if __name__ == "__main__":
    get_mod_params = Register_func["resnet50"]
    mod, pra = get_mod_params()
