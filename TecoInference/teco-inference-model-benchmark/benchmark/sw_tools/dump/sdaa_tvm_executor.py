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
import tvm
from tvm import relay
from .executor.base_executor import BaseExecutor
from .utils.modules import load_module_from_file


class SdaaTvmExecutor(BaseExecutor):

    def init(self, init_configs=None):
        self.init_configs = init_configs

    def load_model(self, model, load_model_configs=None):
        self.load_model_configs = load_model_configs
        self.dtype = load_model_configs.get("dtype")
        self.target = load_model_configs.get("target")
        self.device_type = load_model_configs.get("device_type")
        self.passes = load_model_configs.get("passes")
        self.disabled_pass = load_model_configs.get("disabled_pass")
        self.opt_level = load_model_configs.get("opt_level")
        self.build_config = load_model_configs.get("build_config")
        self.print_ir = load_model_configs.get("print_ir")

        self.ir_module, self.params = \
            relay.frontend.from_onnx(model, dtype=self.dtype)

        # passes
        if self.passes is not None:
            passes_module = load_module_from_file(self.passes)
            self.ir_module = passes_module.use_passes(self.ir_module, load_model_configs)
        if True is self.print_ir:
            print("self.ir_module: ", self.ir_module)
        # build
        target = tvm.target.Target(self.target)
        with tvm.transform.PassContext(opt_level=0,
                                       disabled_pass=self.disabled_pass,
                                       config=self.build_config):
            lib = relay.build(self.ir_module, target=target, params=self.params)

        # engine
        self.engine = tvm.runtime.create_engine(lib, self.device_type)
        self.context = self.engine.create_context()

    def execute(self, output_names, inputs, execute_configs=None):
        self.execute_configs = execute_configs
        self.use_device_id = execute_configs.get("use_device_id")

        if self.use_device_id is None:
            self.devices = [tvm.device(self.device_type)]
        elif isinstance(self.use_device_id, list):
            self.devices = [tvm.device(self.device_type, device_id) \
                            for device_id in self.use_device_id]
        else:
            self.devices = [tvm.device(self.device_type, self.use_device_id)]

        for dev in self.devices:
            for key, value in inputs.items():
                self.context.set_input(key=key, value=value, dev=dev)

            self.context.executor_run(dev=dev)

        output_num = self.context.get_num_outputs()

        outputs_devices = []
        for dev in self.devices:
            outputs_dev = []
            for i in range(output_num):
                outputs_dev.append(self.context.get_output(i, dev))

            outputs_devices.append(outputs_dev)

        # all devices use same input, return output of device 0 as model output
        return outputs_devices[0]

    def release(self):
        self.context.release()
        self.engine.release()
