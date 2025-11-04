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
import os
import sys
import time
import pathlib
import importlib.util
from pathlib import Path
PASS_PATH = Path(__file__).resolve().parent.parent / "teco-inference-model-benchmark/benchmark/pass"

sys.path.append(str(Path(__file__).resolve().parent.parent / "teco-inference-model-benchmark/benchmark"))
from networks.pass_path import pass_path as get_pass_path

class Engine():
    def __init__(self, 
                 task=None,
                 model=None,
                 backend=None,
                 ckpt=None,
                 pass_path=None,
                 model_name=None,
                 batch_size=None,
                 *args,
                 **kwargs):
        self.ckpt = ckpt
        self.backend = backend
        self.model_name = model_name
        self.pass_path = pass_path if pass_path is not None else get_pass_path(model_name=model_name, bs=batch_size)
        if self.ckpt.endswith("onnx"):
            if self.backend == None or self.backend:
                self.backend == "tvm"
            else:
                NotImplementedError("if checkpoint is onnx, backend must be tvm")
        else:
            NotImplementedError("only support onnx checkpoint")

        self._init_model(self, *args, **kwargs)

        self._forward_params = self._sanitize_parameters(**kwargs)

    def _init_model(self, *args, **kwargs):
        if self.backend == "tvm":
            self._init_tvm_model(self, *args, **kwargs)
        elif self.backend == "trt":
            self._init_trt_model(self, *args, **kwargs)
        else:
            NotImplementedError("only support tvm backend")

    
    def timer_decorator(decorator_arg):
        def decorator(func):
            def wrapper(self,*args, **kwargs):
                start_time = time.time()
                result = func(self,*args, **kwargs)
                end_time = time.time()
                if decorator_arg == "e2e":
                    self.e2e = end_time - start_time
                elif decorator_arg in {"tvm_run", "trt_run", "onnxrt_run"}:
                    self.run_time = end_time - start_time
                elif decorator_arg == 'preprocess':
                    self.pre_time = end_time - start_time
                return result
            return wrapper

        return decorator
    
    
    @abstractmethod
    def _init_tvm_model(self, *args, **kwargs):
        '''编译tvm模型'''
        raise NotImplementedError("_init_tvm_model not implemented")
    
    @abstractmethod
    def _init_trt_model(self, *args, **kwargs):
        '''编译trt模型'''
        raise NotImplementedError("_init_trt_model not implemented")

    @timer_decorator('tvm_run')
    def __call__(self, inputs, *args, num_workers=None, batch_size=None, **kwargs):

        forward_params = self._sanitize_parameters(**kwargs)

        forward_params = {**self._forward_params, **forward_params}
        
        return self.forward(inputs, **forward_params)
    
    def forward(self, model_inputs, **forward_params):
        model_outputs = self._forward(model_inputs, **forward_params)
        return model_outputs

    @abstractmethod
    def _sanitize_parameters(self, **parameters):
        """
        _sanitize_parameters will be called with any excessive named arguments from either `__init__` or `__call__`
        methods. It should return 3 dictionnaries of the resolved parameters used by the various `preprocess`,
        `forward` and `postprocess` methods. Do not fill dictionnaries if the caller didn't specify a kwargs. This
        let's you keep defaults in function signatures, which is more "natural".

        It is not meant to be called directly, it will be automatically called and the final parameters resolved by
        `__init__` and `__call__`
        """
        raise NotImplementedError("_sanitize_parameters not implemented")


    @abstractmethod
    def _forward(self, model_inputs, **forward_parameters):
        """
        _forward will receive the prepared dictionary from `preprocess` and run it on the model. This method might
        involve the GPU or the CPU and should be agnostic to it. Isolating this function is the reason for `preprocess`
        and `postprocess` to exist, so that the hot path, this method generally can run as fast as possible.

        It is not meant to be called directly, `forward` is preferred. It is basically the same but contains additional
        code surrounding `_forward` making sure tensors and models are on the same device, disabling the training part
        of the code (leading to faster inference).
        """
        raise NotImplementedError("_forward not implemented")


    def load_module_from_file(self, file_path):
        """
        load module pass form file_path
        """
        file = pathlib.Path(file_path)

        spec = importlib.util.spec_from_file_location(file.stem, file_path)

        module = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(module)

        return module

    def get_shape_dict(self, onnx_model):

        val_info_proto = onnx_model.graph.input

        names = []
        shapes = []
        for val in val_info_proto:
            name = val.name
            tensor_type = val.type.tensor_type
            shape = []
            for dim in tensor_type.shape.dim:
                shape.append(int(dim.dim_value))

            names.append(name)
            shapes.append(shape)
        
        return dict(zip(names, shapes))

    def release(self,):
        print("release memory")
        if self.module:
            self.module.release()
            self.engine.release()