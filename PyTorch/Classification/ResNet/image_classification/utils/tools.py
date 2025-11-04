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
    from module_align import LayerPrecisionCompareTool
except:
    print('module_align is not installed, please install it.')

from torch.profiler import profile, record_function, ProfilerActivity
import torch

def get_layer_precision_tool(model,args):
    tool = LayerPrecisionCompareTool(model_name='res50_ngc_amp',
                                          model=model,
                                          check_backward=True,
                                          compare_dtype=torch.float32,
                                          check_mode=True,
                                          use_json=True,
                                          fd_init=True,
                                          save_ctx=True,
                                          use_amp=True,
                                          save_path=args.save_dir,
                                          )
    return tool

def get_profiler_tool(args):
    activities=[torch.profiler.ProfilerActivity.CPU
                    ]
    if args.device == 'sdaa':
        activities.append(torch.profiler.ProfilerActivity.SDAA)
    elif args.device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    tool = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=1, warmup=6, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                args.profiler_path),
            record_shapes=True,
            # with_stack=True,
        )
    return tool

def get_tools(model,args):
    try:
        module_align_tool = get_layer_precision_tool(model,args) if args.layer_diff else None
    except:
        module_align_tool = None
    
    if args.profiler:
        profiler_tool = get_profiler_tool(args)
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
    