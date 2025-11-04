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

try:
    import tvm
    from tvm import relay
except:
    print("import tvm failed! if test teco inference, please install teco-inference!")

import copy
import onnx
import paddle

import numpy as np
import onnxruntime as rt
from .base import Engine


class TecoInferEngine(Engine):
    def __init__(self,
                task=None,
                model=None,
                ckpt=None,
                target='sdaa',
                dtype='float32',
                half=True,
                save_engine=False,
                batch_size=1,
                input_size=None,
                input_name=None,
                divided=True,
                rank=0,
                *args, **kwargs):
        self.task = task
        self.model = model
        self.model_path = ckpt
        self.target = target
        self.batch_size = batch_size
        self.dtype = dtype
        self.half = half
        self.divided = divided # divide 为 True 则进行单卡四核组推断
        self.save_engine = save_engine
        self.input_name = input_name
        self.rank = rank
        self.input_size = input_size

        if "sdaa" in target or "cpu" in target:
            backend = "tvm"
        elif 'gpu' in target:
            backend = "trt"
            try:
                import tensorrt as trt
                from collections import OrderedDict, namedtuple
            except:
                print("Failed to import TensorRT!")
            paddle.set_device(self.target)
        else:
            backend = "onnx"
            paddle.set_device('cpu')
        super().__init__(task=task, model=model, backend=backend, ckpt=ckpt, batch_size=batch_size, *args, **kwargs)

    def _init_model(self, *args, **kwargs):
        if self.backend == "tvm":
            self._init_tvm_model(self, *args, **kwargs)
        elif self.backend == "trt":
            self._init_trt_model(self, *args, **kwargs)
        elif self.backend == 'onnx':
            self._init_onnx_model(self, *args, **kwargs)
        else:
            NotImplementedError("only support tvm backend and trt backend.")

    def __call__(self, inputs, *args, **kwargs):
        return super().__call__(inputs, *args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}

    def _forward(self, model_inputs, **forward_parameters):
        if self.backend == "tvm":
            if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, list):
                model_inputs = [model_inputs]
            model_inputs = [tvm.runtime.from_numpy(np.ascontiguousarray(model_input)) for model_input in model_inputs]
            if self.divided:
                outputs = self._run_tvm(model_inputs)
                if len(outputs)==1:
                    outputs = outputs[0]
            else:
                outputs = self._mpirun_tvm(np.ascontiguousarray(model_inputs))
            return outputs

        elif self.backend == "trt":
            if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, list):
                model_inputs = [model_inputs]
            model_inputs = [paddle.to_tensor(model_input) for model_input in model_inputs]
            outputs = self._run_trt(model_inputs)
            if len(outputs)==1:
                outputs = outputs[0]
            return outputs

        elif self.backend == 'onnx':
            outputs = self._onnx_run(model_inputs)
            if len(outputs)==1:
                outputs = outputs[0]
            return outputs

        else:
            print("only support tvm and trt backend !")
            raise NotImplemented


    def _init_trt_model(self, *args, **kwargs):
        """
        init trt model from enging or onnx
        """
        try:
            import tensorrt as trt
            from collections import OrderedDict, namedtuple
        except:
            print("Failed to import TensorRT!")
        print(f'Loading {self.ckpt} for TensorRT inference...')
        if self.ckpt.endswith('onnx'):
            # init trt model from onnx
            self.model = self.get_engine(self.ckpt, half=self.half, save=self.save_engine)
        else:
            # init trt model from engine
            logger = trt.Logger(trt.Logger.INFO)
            with open(self.ckpt, 'rb') as f, trt.Runtime(logger) as runtime:
                self.model = runtime.deserialize_cuda_engine(f.read())

        self.input_names = []
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.context = self.model.create_execution_context()
        self.context.active_optimization_profile = 0
        self.bindings = OrderedDict()
        self.output_names = []

        # check half and dynamic
        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if self.model.binding_is_input(i):
                if -1 in tuple(self.model.get_binding_shape(i)):
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.half = True
                    self.input_names.append(name)
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = paddle.to_tensor(np.empty(shape, dtype=dtype))
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def get_engine(self,
                onnx,
                half=False,
                save=True):
        try:
            import tensorrt as trt
            from collections import OrderedDict, namedtuple
        except:
            print("Failed to import TensorRT!")
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        profile = builder.create_optimization_profile()
        # set shape
        if isinstance(self.input_size, list):
            for i in range(len(self.input_size)):
                profile.set_shape(inputs[i].name, self.input_size[i], self.input_size[i], self.input_size[i])
        config.add_optimization_profile(profile)
        for inp in inputs:
            print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

        f = onnx.replace(".onnx", ".engine")
        print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = None
        try:
            engine_bytes = builder.build_serialized_network(network, config)
        except AttributeError:
            engine = builder.build_engine(network, config)
            engine_bytes = engine.serialize()
            del engine
        assert engine_bytes

        with trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        if save:
            print(f"enging saving to {f}")
            with open(f, 'wb') as t:
                t.write(engine_bytes)

        return engine

    @Engine.timer_decorator('trt_run')
    def _run_trt(self, model_inputs):
        for k,v in zip(self.input_names,model_inputs):
            self.binding_addrs[k] = int(v.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        y = [i.numpy() for i in y]
        return y

    def _init_tvm_model(self, *args, **kwargs):
        onnx_model = onnx.load(self.model_path)

        # shape_dict
        input_names = [input.name for input in onnx_model.graph.input]
        input_dims= [input.type.tensor_type.shape.dim for input in onnx_model.graph.input]
        # check dynamic
        if True in ['dim_param' in dim.__str__() for dim in input_dims]:
            shape_dict = dict(zip(input_names, self.input_size))
        else:
            input_shapes=[]
            for input_dim in input_dims:
                input_shapes.append([dim.dim_value for dim in input_dim])
            shape_dict = dict(zip(input_names, input_shapes))

        model, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        if 'sdaa' in self.target:
            passes_module = self.load_module_from_file(self.pass_path)
            model = passes_module.use_passes(model)

            # convert_float_to_float16 kwargs
            need_convert = True
            skip_ops = None
            convert_all = False
            convert_judge = None
            skip_input_name = None
            # if hasattr(passes_module, 'normalized'):
            #     normalized = passes_module.normalized
            # else:
            #     normalized = True
            if hasattr(passes_module, 'need_convert'):
                need_convert = passes_module.need_convert
            if hasattr(passes_module, 'skip_ops'):
                skip_ops = passes_module.skip_ops
            if hasattr(passes_module, 'convert_all'):
                convert_all = passes_module.convert_all
            if hasattr(passes_module, 'convert_judge'):
                convert_judge = passes_module.convert_judge
            if hasattr(passes_module, 'skip_input_name'):
                skip_input_name = passes_module.skip_input_name

            if need_convert and model is not None and self.dtype=='float16':
                if hasattr(passes_module, 'normalized'):
                    normalized = passes_module.normalized
                    model = relay.frontend.convert_float_to_float16(model,
                                                                    skip_ops=skip_ops,
                                                                    convert_all=convert_all,
                                                                    convert_judge=convert_judge,
                                                                    skip_input_name=skip_input_name,
                                                                    normalized=normalized)
                else:
                    model = relay.frontend.convert_float_to_float16(model,
                                                                    skip_ops=skip_ops,
                                                                    convert_all=convert_all,
                                                                    convert_judge=convert_judge,
                                                                    skip_input_name=skip_input_name)

            target = tvm.target.Target("sdaa -libs=tecodnn,tecoblas", host="llvm")
        elif 'cpu' in self.target:
            target = 'llvm'

        with tvm.transform.PassContext(opt_level=0, disabled_pass=["SimplifyInference"]):
            lib = relay.build(model, target=target,params=params)

        self.engine  = tvm.runtime.create_engine(lib,self.target)
        self.module = self.engine.create_context()


    def build_engine(self,ir_module, params, target, device_type, disabled_pass=None, config=None):
        with tvm.transform.PassContext(opt_level=0, disabled_pass=disabled_pass, config=config):
            lib = relay.build(ir_module, target=target, params=params)
        engine = tvm.runtime.create_engine(lib, self.target)
        self.module = engine.create_context()
        return engine

    def get_tvm_engine(self):
        sdaa_target = tvm.target.Target("sdaa --libs=tecodnn,tecoblas", host="llvm")
        sdaa_device_type = "sdaa"
        sdaa_disabled_pass = ["SimplifyInference"]
        return self.build_engine(self.module_tmp,
                                        self.param,
                                        target=sdaa_target,
                                        device_type=sdaa_device_type,
                                        disabled_pass=sdaa_disabled_pass)


    @Engine.timer_decorator('tvm_run')
    def _run_tvm(self, model_inputs,):
        with tvm.transform.PassContext(opt_level=0):
            future = self.module.run_async(model_inputs)
            tvm_output = future.get()
            tvm_output=[i.numpy() for i in tvm_output]
        return tvm_output

    @Engine.timer_decorator('tvm_run')
    def _mpirun_tvm(self, model_inputs,):
        print("==================== mpi run ====================")
        with tvm.transform.PassContext(opt_level=0):
            tvm_output = []
            dev = tvm.device("sdaa", self.rank)
            self.module.set_input(key=self.input_name, value=model_inputs, dev=dev)
            self.module.executor_run(dev=dev)
            tvm_output = self.module.get_output(0, dev)

        return tvm_output

    def extract_onnx_model(self, model_path,
                       input_names,
                       output_names):
        temp_model_path = model_path + f".extract.onnx"

        onnx.utils.extract_model(model_path,
                                temp_model_path,
                                input_names,
                                output_names,
                                False)

        return temp_model_path

    def topological_sorting_onnx_model(self, onnx_model):
        inputs = onnx_model.graph.input
        old_nodes = copy.deepcopy(onnx_model.graph.node)
        new_nodes = copy.deepcopy(old_nodes)
        while len(new_nodes) > 0:
            del new_nodes[0]

        # topological sorting
        input_names = set()
        for _input in inputs:
            input_names.update([_input.name])

        for i in range(len(old_nodes)):
            for old_node in old_nodes:
                flag = True

                for input_name in old_node.input:
                    if not input_name in input_names:
                        flag = False

                if flag:
                    new_nodes.append(old_node)
                    input_names.update(old_node.output)

            for node in new_nodes:
                if node in old_nodes:
                    old_nodes.remove(node)

            if len(old_nodes) == 0:
                break

        while len(onnx_model.graph.node) > 0:
            del onnx_model.graph.node[0]
        for node in new_nodes:
            onnx_model.graph.node.append(node)

        return onnx_model

    def run_opt_pass(self,model, passes, opt_level):
        passes = passes if isinstance(passes, list) else [passes]
        seq = tvm.transform.Sequential(passes)
        with tvm.transform.PassContext(opt_level=opt_level):
            model = seq(model)

        return model


    @Engine.timer_decorator('tvm_run')
    def _onnx_run(self, inputs,  **forward_parameters):
        if not isinstance(inputs,tuple) and not isinstance(inputs,list):
            inputs=[inputs]
        input_names=[input.name for input in self.session.get_inputs()]
        _input={input_name:model_input for input_name,model_input in zip(input_names,inputs)}
        output = self.session.run(output_names=None, input_feed=_input)

        return output

    def _init_onnx_model(self, *args, **kwargs):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = rt.InferenceSession(self.ckpt, sess_options)

    def _as_list(self, file):
        if isinstance(file, str):
            return [file,]
        elif isinstance(file, (list,tuple)):
            return file
        elif isinstance(file, dict):
            return [file]
        else:
            raise ValueError(f"wrong type of file name, except str or list, but got {type(file)}.")

    def release(self,):
        print("release memory")
        if self.module:
            self.engine.release()
            self.module.release()
