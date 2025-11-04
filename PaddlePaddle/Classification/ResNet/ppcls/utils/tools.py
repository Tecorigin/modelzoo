# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted to tecorigin hardware

try:
    from paddle_module_align import PaddleModuleAlignTools
except:
    print('module_align is not installed, please install it.')

import paddle.profiler as profiler

def get_layer_precision_tool(engine):
    tool = PaddleModuleAlignTools(
                        "toy_model",
                        engine.model,
                        check_backward=True,
                        check_mode=True,
                        fd_init=True,
                        save_ctx=True,
                        use_amp=True,
                        save_path =engine.config['Global']['precision_align_path'],
                    )
    return tool

def get_profiler_tool(engine):
    def my_on_trace_ready(prof): # 定义回调函数，性能分析器结束采集数据时会被调用
        callback = profiler.export_chrome_tracing(engine.config["Global"]['profiler_path']) # 创建导出性能数据到 profiler_demo 文件夹的回调函数
        callback(prof)  # 执行该导出函数
        prof.summary(sorted_by=profiler.SortedKeys.GPUTotal) # 打印表单，按 GPUTotal 排序表单项

    tool = profiler.Profiler(#targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                        on_trace_ready=my_on_trace_ready, 
                        scheduler = (10, 16), 
                        timer_only=False, 
                        record_shapes =True) 
    return tool

def get_tools(engine):
    
    module_align_tool = get_layer_precision_tool(engine) if engine.config['Global']['precision_align'] else None
    
    if engine.config["Global"]['profiler']:
        profiler_tool = get_profiler_tool(engine)
        profiler_tool.start()
    else:
        profiler_tool = None
    
    return module_align_tool,profiler_tool


def t_step(module_align_tool,profiler_tool):
    if module_align_tool:
        module_align_tool.step()
    if profiler_tool:
        profiler_tool.step()

def t_stop(module_align_tool,profiler_tool):
    if module_align_tool:
        module_align_tool.stop()
    if profiler_tool:
        profiler_tool.stop()
    