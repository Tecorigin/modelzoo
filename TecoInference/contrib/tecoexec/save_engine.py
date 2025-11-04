from argparse import ArgumentParser
import os
from pathlib import Path
import onnx
from tvm import relay
import importlib.util
import tvm
from tvm.relay.transform import InferType
import sys
print(str(Path(__file__).resolve().parents[0]))
sys.path.append(str(Path(__file__).resolve().parents[0]))
import numpy as np

def load_module_from_file(path):
    file = Path(path)
    spec = importlib.util.spec_from_file_location(file.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def gen_network_inputs_shape_from_ir_module(ir_module):
    inputs = []
    for param in ir_module["main"].params:
        shape = [int(x) for x in param.checked_type.shape]
        inputs.append(shape)
    return inputs

# topological_sorting will modify onnx_model
def topological_sorting(onnx_model: onnx.ModelProto):
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

    # Note: this logic only applies to top level graph
    # since a sub graph could use intializer from parent graph
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
        raise RuntimeError(f"""Graph is not a DAG: end={end},
            len(graph.node)={len(graph.node)},
            graph.node[end]={graph.node[end]}""")

    graph.ClearField("node")
    graph.node.extend(sorted_nodes)

class SaveEengine:
    def __init__(self,
                 onnx_path,
                 dtype='float16',
                 pass_path=None,
                 save_path=None,
                 topsort=True,
                 input_shapes=None):
        
        self.onnx_path = onnx_path
        self.pass_path = pass_path
        self.save_path = save_path if save_path else os.path.join('./', os.path.basename(onnx_path)+'_.tecoengine')
        self.pass_path = pass_path if pass_path else str(Path(__file__).resolve().parents[0].joinpath('pass/default_pass.py'))
        self.dtype = dtype
        self.toposort = topsort
        self.input_shapes = input_shapes
        self.shape_dict = None

    def _pass_process(self):
        assert os.path.exists(self.onnx_path)
        onnx_model = onnx.load(self.onnx_path)
        if self.toposort:
            topological_sorting(onnx_model)

        # dynamic
        if self.input_shapes is not None:
            self.input_names = [input.name for input in onnx_model.graph.input]
            self.shape_dict = dict(zip(self.input_names, self.input_shapes))
        print(f"buliding shape_dict={self.shape_dict}")
        ir_module, param = relay.frontend.from_onnx(onnx_model, shape=self.shape_dict, dtype=self.dtype)
        passes_module = load_module_from_file(self.pass_path)
        ir_module = passes_module.use_passes(ir_module)

        need_convert = True
        skip_ops = None
        convert_all = False
        convert_judge = None
        skip_input_name = None
        normalized = None

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
        if hasattr(passes_module, 'normalized'):
            normalized = passes_module.normalized

        return ir_module, param, need_convert, skip_ops, convert_all, convert_judge, skip_input_name, normalized

    def _build_engine(self, ir_module, params, target, device_type, disabled_pass=None, config=None):
        with tvm.transform.PassContext(opt_level=0, disabled_pass=disabled_pass, config=config):
            lib = relay.build(ir_module, target=target, params=params)
        engine = tvm.runtime.create_engine(lib, device_type)

        if self.input_shapes is None:
            self.input_shapes = gen_network_inputs_shape_from_ir_module(ir_module)
        print("")
        return engine, lib

    def _save_engine(self, engine, path:str):
        tvm.runtime.save_engine(engine, path)
        return path    
    
    def save(self):
        ir_module, param, need_convert, skip_ops, convert_all, convert_judge, skip_input_name, normalized = self._pass_process()

        if ir_module is not None and hasattr(self, 'dtype') and self.dtype=='float16' and need_convert:
            print(f"\n {skip_ops=}, {convert_judge=}, {convert_all=}, {skip_input_name=}, {normalized=}")
            if normalized is not None:
                ir_module = relay.frontend.convert_float_to_float16(ir_module, skip_ops=skip_ops, convert_judge=convert_judge, convert_all=convert_all, skip_input_name=skip_input_name, normalized=normalized)
            else:
                ir_module = relay.frontend.convert_float_to_float16(ir_module, skip_ops=skip_ops, convert_judge=convert_judge, convert_all=convert_all, skip_input_name=skip_input_name)
            ir_module = InferType()(ir_module)
        
        sdaa_target = tvm.target.Target("sdaa --libs=tecodnn,tecoblas", host="llvm")
        sdaa_device_type = "sdaa"
        sdaa_disabled_pass = ["SimplifyInference"]
        sdaa_engine, lib = self._build_engine(ir_module,
                                param,
                                target=sdaa_target,
                                device_type=sdaa_device_type,
                                disabled_pass=sdaa_disabled_pass)
    
        path = self._save_engine(lib, self.save_path)
        print(f'Finished! Engine save in {path}!, input_shapes:{self.input_shapes}')
        return self.input_shapes
'''
Example:
    python save_eingine.py -op xxx.onnx
        Param:
            --onnx_path/-op onnx路径
            --pass_path/-pp pass_path 路径
            --save_path/-sp 保存路径如：./resnet50.tecoengine，默认./xxx.onnx.tecoengine
            --onnx_dtype onnx dtype, 默认 float16 

'''        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--onnx_path', '-op', required=True)
    parser.add_argument('--pass_path', '-pp', required=True)
    parser.add_argument('--save_path', '-sp', default=None)
    parser.add_argument('--onnx_dtype', default='float16')
    args = parser.parse_args()
    
    SaveEengine(onnx_path=args.onnx_path,
                pass_path=args.pass_path,
                save_path=args.save_path,
                dtype=args.onnx_dtype).save()

