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
    import tvm.relay as relay
    import tvm
except:
    print("import tvm failed! if test teco inference, please install teco-inference!")

import torch
import onnx
import onnxruntime as rt
import numpy as np
from .base import Engine


class TecoInferEngine(Engine):
    def __init__(self,
                 task=None,
                 model=None,
                 backend=None,
                 ckpt=None,
                 batch_size=None,
                 input_size=None,
                 input_name="",
                 target='sdaa',
                 dtype='float32',
                 save_engine=False,
                 devide=True,
                 rank=0,
                 *args,
                 **kwargs):
        self.model_path = ckpt
        self.batch_size = batch_size
        self.input_size = input_size
        self.input_name = input_name
        self.target = target
        self.dtype = dtype
        self.save_engine = save_engine
        self.devide = devide
        self.rank = rank

        if "sdaa" in target or "cpu" in target:
            backend = "tvm"
            self.dev = tvm.device(self.target, self.rank)
        elif 'onnx' in target:
            print("use onnx")
            backend = "onnxruntime"
            self.device = torch.device("cpu")
        else:
            backend = "trt"
            try:
                import tensorrt as trt
                from collections import OrderedDict, namedtuple
            except:
                print("Failed to import TensorRT!")
            self.device = torch.device(self.target)
        super().__init__(task=task, model=model, backend=backend, ckpt=ckpt, batch_size=batch_size, *args, **kwargs)


    def __call__(self,inputs, *args, **kwargs):
        return super().__call__(inputs, *args, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        return {}

    def _forward(self, model_inputs, **forward_parameters):
        if self.backend == "onnxruntime":
            if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, list):
                model_inputs = [model_inputs]
            model_inputs = {input_name:model_input for input_name,model_input in zip(self.input_names,model_inputs)}

            outputs = self.onnx_run(model_inputs)
            if len(outputs)==1:
                outputs=outputs[0]
            return outputs
        if self.backend == "tvm":
            if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, list):
                model_inputs = [model_inputs]
            model_inputs = [tvm.runtime.from_numpy(np.ascontiguousarray(model_input)) for model_input in model_inputs]
            if self.devide:
                outputs = self._run(model_inputs)
                if len(outputs)==1:
                    outputs=outputs[0]
            else:
                outputs = self.mpi_run(np.ascontiguousarray(model_inputs))

        elif self.backend == "trt":
            if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, list):
                model_inputs = [model_inputs]
            model_inputs = [torch.from_numpy(model_input).to(self.device) for model_input in model_inputs]
            outputs = self._run_trt(model_inputs)
            if len(outputs)==1:
                outputs=outputs[0]
        else:
            print("only support tvm and trt backend !")
            raise NotImplemented
        return outputs

    @Engine.timer_decorator('tvm_run')
    def onnx_run(self,inputs):
        return self.session.run(output_names=None, input_feed=inputs)

    def _init_model(self, *args, **kwargs):
        if self.backend == 'onnxruntime':
            self._init_onnxruntime(self, *args, **kwargs)
        elif self.backend == "tvm":
            self._init_tvm_model(self, *args, **kwargs)
        elif self.backend == "trt":
            self._init_trt_model(self, *args, **kwargs)
        else:
            NotImplementedError("not supported backend")

    def _init_onnxruntime(self, *args, **kwargs):
        providers = ['CPUExecutionProvider']
        self.session = rt.InferenceSession(self.model_path, providers=providers)
        self.input_names = [input.name for input in self.session.get_inputs()]

    def _init_trt_model(self, *args, **kwargs):
        """
        init trt model from enging or onnx
        """
        try:
            import tensorrt as trt
            from collections import OrderedDict, namedtuple
        except:
            print("Failed to import TensorRT!")
        print(f'Loading {self.model_path} for TensorRT inference...')
        self.half = True if self.dtype == 'float16' else False
        if (self.model_path.endswith('onnx')):
            # init trt model from onnx
            self.model = self.get_engine(self.model_path, half=self.half, save=self.save_engine)
        else:
            # init trt model from engine
            logger = trt.Logger(trt.Logger.INFO)
            with open(self.model_path, 'rb') as f, trt.Runtime(logger) as runtime:
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
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())

    def _init_tvm_model(self, *args, **kwargs):
        onnx_model = onnx.load(self.model_path)
        if "bert" not in self.input_name:
            graph_topological_sort(onnx_model)

        # shape_dict
        self.input_names = [input.name for input in onnx_model.graph.input]
        input_dims= [input.type.tensor_type.shape.dim for input in onnx_model.graph.input]
        # check dynamic
        if True in ['dim_param' in dim.__str__() for dim in input_dims]:
            shape_dict = dict(zip(self.input_names, self.input_size))
        else:
            input_shapes=[]
            for input_dim in input_dims:
                input_shapes.append([dim.dim_value for dim in input_dim])
            shape_dict = dict(zip(self.input_names, input_shapes))

        print(f"{shape_dict=}")
        

        if 'sdaa' in self.target:
            passes_module = self.load_module_from_file(self.pass_path)
            if hasattr(passes_module, 'specified_op_dtype_dict'):
                specified_op_dtype_dict=passes_module.specified_op_dtype_dict
                model, params = relay.frontend.from_onnx(
                    onnx_model, 
                    shape_dict, 
                    specified_op_dtype=specified_op_dtype_dict
                    )
            else:
                model, params = relay.frontend.from_onnx(
                    onnx_model, 
                    shape_dict, 
                    )
            model = passes_module.use_passes(model)

            # convert_float_to_float16 kwargs
            need_convert = True
            skip_ops = None
            convert_all = False
            convert_judge = None
            skip_input_name = None
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
            model, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            target = 'llvm'

        with tvm.transform.PassContext(opt_level=0, disabled_pass=["SimplifyInference"]):
            lib = relay.build(model, target=target,params=params)

        self.engine  = tvm.runtime.create_engine(lib,self.target)
        self.module = self.engine.create_context()


    @Engine.timer_decorator('tvm_run')
    def mpi_run(self, input, *args, **kwargs):
        '''运行模型'''
        assert hasattr(self, 'module')
        self.module.set_input(key=self.input_name, value=input, dev=self.dev)
        self.module.executor_run(dev=self.dev)
        tvm_output=self.module.get_output(0, self.dev)

        return tvm_output

    @Engine.timer_decorator('tvm_run')
    def _run(self, input, *args, **kwargs):
        '''运行模型'''
        assert hasattr(self, 'module')
        future = self.module.run_async(input)
        tvm_output=future.get()
        tvm_output=[i.numpy() for i in tvm_output]
        return tvm_output


    @Engine.timer_decorator('trt_run')
    def _run_trt(self, model_inputs):
        for k,v in zip(self.input_names,model_inputs):
            self.binding_addrs[k] = int(v.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)]
        y = [i.cpu().float() for i in y]
        return y

    def get_engine(self,
                    onnx_path,
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
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')
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
        f = onnx_path.replace(".onnx", ".engine")
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

def graph_topological_sort(onnx_model: onnx.ModelProto):
    graph = onnx_model.graph
    deps_count = [0] * len(graph.node)  # dependency count of each node
    deps_to_nodes = {}  # input to node indice
    sorted_nodes = []  # initialize sorted_nodes
    for node_idx, node in enumerate(graph.node):
        # CANNOT use len(node.input) directly because input can be optional
        deps_count[node_idx] = sum(1 for _ in node.input if _)
        if deps_count[node_idx] == 0:  # Constant doesn't depend on any inputs
            sorted_nodes.append(graph.node[node_idx])
            continue

        for input_name in node.input:
            if input_name not in deps_to_nodes:
                deps_to_nodes[input_name] = [node_idx]
            else:
                deps_to_nodes[input_name].append(node_idx)

    # Note: this logic only applies to top level graph since a sub graph could use intializer from parent graph
    initializer_names = [init.name for init in graph.initializer]
    graph_input_names = [input_.name for input_ in graph.input]
    input_names = initializer_names + graph_input_names
    input_names.sort()
    prev_input_name = None
    for input_name in input_names:
        if prev_input_name == input_name:
            continue

        prev_input_name = input_name
        if input_name in deps_to_nodes:
            for node_idx in deps_to_nodes[input_name]:
                deps_count[node_idx] = deps_count[node_idx] - 1
                if deps_count[node_idx] == 0:
                    sorted_nodes.append(graph.node[node_idx])

    start = 0
    end = len(sorted_nodes)

    while start < end:
        for output in sorted_nodes[start].output:
            if output in deps_to_nodes:
                for node_idx in deps_to_nodes[output]:
                    deps_count[node_idx] = deps_count[node_idx] - 1
                    if deps_count[node_idx] == 0:
                        sorted_nodes.append(graph.node[node_idx])
                        end = end + 1
        start = start + 1

    if end != len(graph.node):
        raise RuntimeError(
            f"Graph is not a DAG: end={end}, len(graph.node)={len(graph.node)}, graph.node[end]={graph.node[end]}"
        )

    graph.ClearField("node")
    graph.node.extend(sorted_nodes)
