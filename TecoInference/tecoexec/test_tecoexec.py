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
import os
from pathlib import Path
import sys
import time
from typing import Dict
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[0]))
from save_engine import SaveEengine, onnx
from benchmark_tools.teco_unittest import TecoUnittest as TestCase
from benchmark_tools.teco_unittest import TecoTestsMetaCommon
from benchmark_tools.common.wrapper import run_table
from benchmark_tools.common.utils import load_yaml
import shutil
import subprocess
import re

@run_table({
    'framework':['TecoInference'],
    'task':['inference']
})
class TestTecoexec(TestCase, metaclass=TecoTestsMetaCommon):
    task_label='TestPerf'
    model_name_list = os.getenv('TECOEXEC_CONFIGS', str(Path(__file__).resolve().parents[0] / 'testcase_configs/tecoexec_config.yaml'))
    MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))

    @classmethod
    def setUpClass(cls):
        TestCase.setUpClass()
        print('setUpClass...')
        OUTPUT_DIR = os.getenv('OUTPUT_DIR', time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))
        cls.output_dir = str(Path(__file__).resolve().parents[0] / f"test_tecoexec_logs/{OUTPUT_DIR}/")
        shutil.rmtree(cls.output_dir, ignore_errors=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        if not hasattr(cls, 'model_configs'):
            cls.model_configs = load_yaml(cls.model_name_list)

    def init_vars_by_dict(self, vars:dict):
        '''给实例赋值'''
        # support 3core
        if self.MAX_ENGINE_NUMS == 3:
            if isinstance(vars["batch_size"], list):    # dynamic
                vars["batch_size"] = [max(int(bs / 4 * 3), 1) for bs in vars["batch_size"]]
            else:
                vars["batch_size"] = max(int(vars["batch_size"] / 4 * 3), 1)
                vars["input_shapes"] = ",".join([f"{int(s.split('*')[0]) * 3 // 4}{s[s.index('*'):]}" for s in vars["input_shapes"].split(',')])

        for k, v in vars.items():
            setattr(self, k, v)
            print(f'{self}::setattr:{k}={v}')

    def _init_vars(self, testcase_name):
        self.init_vars_by_dict(vars=self.model_configs[testcase_name])
        self.tecoengine_path = os.path.join(self.output_dir, os.path.basename(self.onnx_path)+'.tecoengine')
        if not hasattr(self, 'onnx_dtype'):
            self.onnx_dtype = 'float16'
        self.rm_tecoengine = True
        self.silent = False
        self.testcase_name = testcase_name

        self.tecoexec_path = os.path.join(os.getenv('TECOEXEC_PATH', './'), 'tecoexec')
        self.log_path = os.path.join(self.output_dir, f"{self.testcase_name}.log")
        self.fout = open(self.log_path, 'w')
        # self.pass_path = str(Path(__file__).resolve().parents[1] / f"teco-inference-model-benchmark/benchmark/pass/{self.pass_path}")


    @classmethod
    def tearDownClass(cls):
        pass

    def check_success(self, proc:subprocess.Popen):
        stdout, stderr = proc.communicate()
        output = stdout.decode()
        print(f"stdout:{output}, stderr:{stderr}")
        self.fout.write(output)
        self.fout.flush()
        average_inference_res = re.findall(r'average inference time is (\s*\d+\.?\d*) ms.', output)
        assert len(average_inference_res) == 1
        latency_res = re.findall(r'latency is (\s*\d+\.?\d*) ms.', output)
        assert len(latency_res) == 1
        avg_inference_time = float(average_inference_res[0])
        latency_time = float(latency_res[0])

        if proc.returncode != 0:
            raise RuntimeError('Server Failed!')
        else:
            return avg_inference_time, latency_time

    def run_cmd(self, ):
        args = [self.tecoexec_path]
        args.append('--loadEngine=' + self.tecoengine_path)
        args.append('--input_shapes=' + self.input_shapes)
        args.append('--fp16')
        args.append('--warmUp=' + str(self.warm_up))
        args.append('--iterations=' + str(self.iterations))
        if hasattr(self, 'input_dtype'):
            args.append('--input_dtype=' + str(self.input_dtype))
        if self.run_sync:
            args.append('--runSync')
        exec_cmd = ' '.join(args)
        try:
            print(exec_cmd)
            proc = subprocess.Popen(exec_cmd, shell=True, close_fds=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.avg_inference_time, self.latency_time = self.check_success(proc)
        except Exception as e:
            print(e)
        print(f"avg_inference_times:{self.avg_inference_time}, latency_times:{self.latency_time}")
        if self.avg_inference_time < 1e-8:
            self.throughput = 0.0
        else:
            self.throughput = self.bs / (self.avg_inference_time / 1000.0)

        output = f"{self.testcase_name}, batch_size:{self.bs}, latency:{self.latency_time} ms, average inference time:{self.avg_inference_time} ms, throughput:{self.throughput}"
        self.fout.write(output)
        self.fout.flush()
        print(output)

    def model_inference(self, testcase_name, *args, **kwargs) -> Dict:
        self._init_vars(testcase_name=testcase_name)
        print(f"onnx_path:{self.onnx_path}, pass_path:{self.pass_path}")

        # check dynamic
        onnx_model = onnx.load(self.onnx_path)
        input_dims= [input.type.tensor_type.shape.dim for input in onnx_model.graph.input]
        if True in ['dim_param' in dim.__str__() for dim in input_dims]:
            if not isinstance(self.batch_size, list):
                self.batch_size = [int(self.batch_size)]
            for bs in self.batch_size:
                self.bs = bs
                onnx_bs = max(self.bs // self.MAX_ENGINE_NUMS, 1)
                self.tecoengine_path = os.path.join(self.output_dir, os.path.basename(self.onnx_path) + f'_bs{onnx_bs}.tecoengine')

                shapes_tecoexec = []
                shapes_engine = []
                input_shape_list = self.input_shapes.split(',')
                for i in range(len(input_shape_list)):
                    if hasattr(self, "batch_index") and i not in self.batch_index:
                        shapes_tecoexec.append(input_shape_list[i])
                        shapes_engine.append(input_shape_list[i])
                    else:
                        shapes_tecoexec.append(f"{onnx_bs}" if '*' not in input_shape_list[i] else f"{onnx_bs}{input_shape_list[i][input_shape_list[i].index('*'):]}")
                        shapes_engine.append(f"{onnx_bs}" if '*' not in input_shape_list[i] else f"{onnx_bs}{input_shape_list[i][input_shape_list[i].index('*'):]}")
                self.input_shapes = ",".join(shapes_tecoexec)    # for tecoexec
                self.input_shapes_ = [eval(f"[{s.replace('*', ',')}]") for s in shapes_engine] # for build engine

                _ = SaveEengine(onnx_path=self.onnx_path,
                                   pass_path=self.pass_path,
                                   save_path=self.tecoengine_path,
                                   dtype=self.onnx_dtype,
                                   input_shapes=self.input_shapes_).save()
                assert os.path.exists(self.tecoengine_path), f"{self.tecoengine_path} Generation failed"
                assert os.path.exists(self.tecoexec_path), f"{self.tecoexec_path} not exists"

                self.run_cmd()
        else:
            self.bs = self.batch_size
            _ = SaveEengine(onnx_path=self.onnx_path,
                                   pass_path=self.pass_path,
                                   save_path=self.tecoengine_path,
                                   dtype=self.onnx_dtype).save()

            assert os.path.exists(self.tecoengine_path), f"{self.tecoengine_path} Generation failed"
            assert os.path.exists(self.tecoexec_path), f"{self.tecoexec_path} not exists"

            self.run_cmd()

        if self.rm_tecoengine:
            os.remove(self.tecoengine_path)
        self.fout.close()
