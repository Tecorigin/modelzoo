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

import time
import collections

import numpy as np

import tvm
from tvm import relay
import tecoinference

from .random import RandomBoundary
import os
from pathlib import Path

def serialize_ir_module_str(ir_module, output_ir_module_path):
    with open(output_ir_module_path, "w") as f:
        f.write(ir_module.astext())


def deserialize_ir_module_str(ir_module_path):
    ir_module = tvm.IRModule()
    ir_module._import(ir_module_path)

    return ir_module


def serialize_ir_module_json(ir_module, output_ir_module_path):
    with open(output_ir_module_path, "w") as f:
        f.write(tvm.ir.save_json(ir_module))


def deserialize_ir_module_json(ir_module_path):
    with open(ir_module_path, "r") as f:
        ir_module = tvm.ir.load_json(f.read())

    return ir_module


def serialize_params(params, output_params_path):
    with open(output_params_path, "wb") as f:
        f.write(relay.save_param_dict(params))


def deserialize_params(params_path):
    with open(params_path, "rb") as f:
        params = relay.load_param_dict(f.read())

    return params


def gen_network_inputs(module, boundary=None, seed=42, onnx_model=None, use_cross_memory=False):
    np.random.seed(seed)

    if boundary is None:
        boundary = RandomBoundary()
    if isinstance(module, tvm.runtime.engine.Engine):
        engine_ctx = module.create_context(enable_cross_memory=use_cross_memory)
        inputs = gen_network_inputs_from_engine(engine_ctx, boundary, onnx_model=onnx_model)
        engine_ctx.release()
    else:
        inputs = gen_network_inputs_from_ir_module(module, boundary)

    return inputs

def gen_network_inputs_from_ir_module(ir_module, boundary):
    inputs = collections.OrderedDict()
    for param in ir_module["main"].params:
        shape = [int(x) for x in param.checked_type.shape]

        dtype = param.checked_type.dtype
        boundary_i = boundary.get(param.name_hint, dtype)

        print("ir_module input name {}, shape {}, dtype {}, boundary [{}, {})".format(
            param.name_hint, shape, dtype, boundary_i.low, boundary_i.high))

        if dtype == "float16" or \
           dtype == "float32" or \
           dtype == "float64":
            random_array = np.random.uniform(boundary_i.low, boundary_i.high,
                                             size=tuple(shape)).astype(dtype)
        elif dtype == "int8" or \
             dtype == "int16" or \
             dtype == "int32" or \
             dtype == "int64":
            random_array = np.random.randint(boundary_i.low, boundary_i.high,
                                             size=tuple(shape)).astype(dtype)
        else:
            raise ValueError("unsupport dtype {}".format(dtype))

        inputs[param.name_hint] = random_array

    return inputs

def gen_network_inputs_from_engine(ctx: tvm.runtime.engine.Context, boundary, onnx_model):
        info = ctx._get_input_info()
        shape = info["shape"]
        dtype = info["dtype"]

        inputs = collections.OrderedDict()
        input_names = [x.name for x in onnx_model.graph.input]
        for inp in input_names:
            inputs[inp] = None # init inputs

        for nm, shp in shape.items():
            assert nm in dtype, "{} not in dtype dict" .format(nm)
            if nm not in input_names:
                continue
            dt = dtype[nm]
            boundary_i = boundary.get(nm, dt)

            if "float" in dt:
                random_array = np.random.uniform(boundary_i.low, boundary_i.high,
                                            size=tuple(shp)).astype(dt)
            elif "uint1" in dt:
                random_array = np.random.randint(boundary_i.low, boundary_i.high,
                                                size=tuple(shp)).astype(bool)
            elif "int" in dt:
                random_array = np.random.randint(boundary_i.low, boundary_i.high,
                                                size=tuple(shp)).astype(dt)
            else:
                raise ValueError("unsupport dtype {}".format(dt))

            inputs[nm] = random_array

        return inputs

def run_pass(ir_module, target, passes=None, opt_level=3):
    with tvm.target.Target(target):
        if passes is not None:
            seq = tvm.transform.Sequential(passes)

            with tvm.transform.PassContext(opt_level=opt_level):
                ir_module = seq(ir_module)

    return ir_module


def build_engine(ir_module, params, target, device_type, disabled_pass=None, config=None):
    with tvm.transform.PassContext(opt_level=0, disabled_pass=disabled_pass, config=config):
        lib = relay.build(ir_module, target=target, params=params)
    engine = tvm.runtime.create_engine(lib, device_type)
    return engine, lib

# pylint: disable=unused-argument
def run_engine_async(engine, inputs, device, run_loops=1, warm_up=0):
    for _ in range(warm_up):
        input_list = []
        for _, data in inputs.items():
            input_tensor = tvm.runtime.from_numpy(data)
            input_list.append(input_tensor)
        future = engine.run_async(input_list)
        output_num = engine.get_num_outputs()
        outs = future.get()
        outputs = [outs[i].numpy() for i in range(output_num)]

    start = time.time()
    for _ in range(run_loops):
        input_list = []
        for _, data in inputs.items():
            input_tensor = tvm.runtime.from_numpy(data)
            input_list.append(input_tensor)
        future = engine.run_async(input_list)
        output_num = engine.get_num_outputs()
        outs = future.get()
        outputs = [outs[i].numpy() for i in range(output_num)]
    infer_time = time.time() - start
    infer_time = infer_time * 1000 / run_loops

    return outputs, 0.0, infer_time, 0.0


def run_engine_executor(engine, inputs, device, run_loops=1, warm_up=0):
    if not isinstance(device, list):
        device = [device]

    # warm up
    for _ in range(warm_up):
        for dev in device:
            for name, data in inputs.items():
                engine.set_input(name, data, dev)
            engine.executor_run(dev=dev)
            dev.sync()
            for i in range(engine.get_num_outputs()):
                engine.get_output(i, dev)

    # run loops
    start = time.time()
    for _ in range(run_loops):
        for dev in device:
            for name, data in inputs.items():
                engine.set_input(name, data, dev)
    h2d_time = time.time() - start
    h2d_time = h2d_time * 1000 / run_loops

    start = time.time()
    for _ in range(run_loops):
        for dev in device:
            engine.executor_run(dev=dev)
        for dev in device:
            dev.sync()
    infer_time = time.time() - start
    infer_time = infer_time * 1000 / run_loops

    outputs = []
    start = time.time()
    for _ in range(run_loops):
        outputs = []
        for dev in device:
            output_tmp = []
            for i in range(engine.get_num_outputs()):
                output_i = engine.get_output(i, dev)
                output_tmp.append(output_i)
            outputs.append(output_tmp)
    d2h_time = time.time() - start
    d2h_time = d2h_time * 1000 / run_loops

    return outputs[0], h2d_time, infer_time, d2h_time

def run_dyn_async(ctx, inputs, device, run_loops=1, warm_up=0):
    for _ in range(warm_up):
        input_list = []
        for _, data in inputs.items():
            input_list.append(data)
        future = ctx.run_async(input_list)
        output_num = ctx.get_num_outputs()
        outs = future.get()
        outputs = [outs[i].numpy() for i in range(output_num)]

    start = time.time()
    for _ in range(run_loops):
        input_list = []
        for _, data in inputs.items():
            input_list.append(data)
        future = ctx.run_async(input_list)
        output_num = ctx.get_num_outputs()
        outs = future.get()
        outputs = [outs[i].numpy() for i in range(output_num)]
    infer_time = time.time() - start
    infer_time = infer_time * 1000 / run_loops

    return outputs, 0.0, infer_time, 0.0    

def run_dyn_sync(ctx, inputs, device, run_loops=1, warm_up=0):
    if not isinstance(device, list):
        device = [device]

    # warm up
    for _ in range(warm_up):
        for dev in device:
            for name, data in inputs.items():
                ctx.set_input(name, data)
            ctx.executor_run()
            dev.sync()
            for i in range(ctx.get_num_outputs()):
                ctx.get_output(i)

    # run loops
    start = time.time()
    for _ in range(run_loops):
        for dev in device:
            for name, data in inputs.items():
                ctx.set_input(name, data)
    h2d_time = time.time() - start
    h2d_time = h2d_time * 1000 / run_loops

    start = time.time()
    for _ in range(run_loops):
        for dev in device:
            ctx.executor_run()
        for dev in device:
            dev.sync()
    infer_time = time.time() - start
    infer_time = infer_time * 1000 / run_loops

    outputs = []
    start = time.time()
    for _ in range(run_loops):
        outputs = []
        for dev in device:
            output_tmp = [ctx.get_output(i) for i in range(ctx.get_num_outputs())]
            outputs.append(output_tmp)
    d2h_time = time.time() - start
    d2h_time = d2h_time * 1000 / run_loops

    return outputs[0], h2d_time, infer_time, d2h_time

def run_engine(engine, inputs, device, run_loops=1, warm_up=0, use_async=False):
    if use_async:
        return run_engine_async(engine, inputs, device, run_loops, warm_up)
    else:
        return run_engine_executor(engine, inputs, device, run_loops, warm_up)

def save_engine(engine, path:str):
    if path.endswith('.onnx'):
        engine_path = os.path.join(get_engine_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path
    tvm.runtime.save_engine(engine, engine_path)
    print(f'Engine save in {engine_path}!')
    return engine_path

def load_engine(path, dev='sdaa'):
    if path.endswith('.onnx'):
        engine_path = os.path.join(get_engine_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path
    engine = tvm.runtime.create_engine(engine_path, dev)
    print(f'Load Engine from {engine_path}!')
    return engine

def get_engine_cache_dir(default_dir=None):
    default_dir = os.path.join(os.environ.get('HOME', str(Path(__file__).resolve().parents[2])), '.cache/teco-inference-engine-cache/')
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

def exist_engine(path:str):
    if path.endswith('.onnx'):
        engine_path = os.path.join(get_engine_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path
    return os.path.exists(engine_path)


def gen_network_inputs_from_given_shape(shapes, dtypes, boundary=None):
    seed = 42
    np.random.seed(seed)

    # dyn runtime only support sync mode, gen one input for one device
    if boundary is None:
        boundary = RandomBoundary()
    inputs = collections.OrderedDict()
    for name in shapes.keys():
        shape = shapes[name]
        dtype = dtypes[name]
        boundary_i = boundary.get(name, dtype)

        print("generate input name {}, shape {}, dtype {}, boundary [{}, {})".format(
            name, shape, dtype, boundary_i.low, boundary_i.high))

        if dtype == "float16" or \
           dtype == "float32" or \
           dtype == "float64":
            random_array = np.random.uniform(boundary_i.low, boundary_i.high,
                                             size=tuple(shape)).astype(dtype)
        elif dtype == "int8" or \
             dtype == "int16" or \
             dtype == "int32" or \
             dtype == "int64":
            random_array = np.random.randint(boundary_i.low, boundary_i.high,
                                             size=tuple(shape)).astype(dtype)
        elif dtype == "uint1" or \
             dtype == "bool":
            random_array = np.random.randint(boundary_i.low, boundary_i.high,
                                            size=tuple(shape)).astype(bool)
        else:
            raise ValueError("unsupport dtype {}".format(dtype))

        inputs[name] = random_array

    return inputs

def get_engine_fbs_cache_dir(default_dir=None):
    default_dir = os.path.join(os.environ.get('HOME', str(Path(__file__).resolve().parents[2])), '.cache/tecoinference-engine-cache/')
    os.makedirs(default_dir, exist_ok=True)
    return default_dir

def gen_engine_name_with_shape(path:str, shapes):
    name, ext = os.path.splitext(os.path.basename(path))
    for _, shape in shapes.items():
        for dim in shape:
            name = name + "_" + str(dim)
    engine_path = os.path.join(get_engine_fbs_cache_dir(), f'{name}{ext}.tecoengine')
    return engine_path

def build_engine_dyn(ir_module, params, target, device_type, disabled_pass=None, config=None, engine_config=None):
    with tvm.transform.PassContext(opt_level=0, disabled_pass=disabled_pass, config=config):
        model_fbs = relay.build(ir_module, target=target, params=params)

    option = tecoinference.EngineOptions(engine_config)
    engine = tecoinference.Engine(bytes(model_fbs), option)

    return engine, model_fbs

def run_dyn_exec(ctx, inputs, device, run_loops=1, warm_up=0, use_async=False):
    if use_async:
        print("INFO: run async mode ! ")
        return run_dyn_async(ctx, inputs, device, run_loops, warm_up)
    else:
        print("INFO: run  mode ! ")
        return run_dyn_sync(ctx, inputs, device, run_loops, warm_up)

def exist_engine_fbs(path:str, shapes=None):
    if path.endswith('.onnx'):
        if shapes:
            engine_path = gen_engine_name_with_shape(path, shapes)
        else:
            engine_path = os.path.join(get_engine_fbs_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path
    return os.path.exists(engine_path)

def save_engine_fbs(engine, path:str, shapes=None):
    if path.endswith('.onnx'):
        if shapes:
            engine_path = gen_engine_name_with_shape(path, shapes)
        else:
            engine_path = os.path.join(get_engine_fbs_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path
    tvm.runtime.save_engine(engine, engine_path)
    print(f'Save TecoInference Engine in {engine_path}')
    return engine_path

def load_engine_fbs(path:str, shapes=None, engine_config=None):
    if path.endswith('.onnx'):
        if shapes:
            engine_path = gen_engine_name_with_shape(path, shapes)
        else:
            engine_path = os.path.join(get_engine_fbs_cache_dir(), f'{os.path.basename(path)}.tecoengine')
    else:
        engine_path = path

    option = tecoinference.EngineOptions(engine_config)
    engine = tecoinference.Engine(engine_path, option)

    print(f'Load TecoInference Engine from {engine_path}')
    return engine
