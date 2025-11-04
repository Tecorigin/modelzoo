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
from .accuracy import assert_rms

from .random import RandomBoundary
from .random import json_dict_to_random_boundary

from .onnx_utils import topological_sorting
from .onnx_utils import extract_model
from .onnx_utils import get_ort_input_names
from .onnx_utils import get_ort_output_names
from .onnx_utils import get_ort_inputs_from_sdaa
from .onnx_utils import get_input_np_dtype_from_onnx
from .onnx_utils import get_input_np_shape_from_onnx

from .relay_utils import serialize_ir_module_str
from .relay_utils import deserialize_ir_module_str
from .relay_utils import serialize_ir_module_json
from .relay_utils import deserialize_ir_module_json
from .relay_utils import serialize_params
from .relay_utils import deserialize_params
from .relay_utils import gen_network_inputs
from .relay_utils import run_pass
from .relay_utils import build_engine
from .relay_utils import run_engine
from .relay_utils import exist_engine
from .relay_utils import save_engine
from .relay_utils import load_engine
from .module import load_module_from_file

from .relay_utils import gen_network_inputs_from_given_shape
from .relay_utils import build_engine_dyn
from .relay_utils import run_dyn_exec
from .relay_utils import exist_engine_fbs
from .relay_utils import save_engine_fbs
from .relay_utils import load_engine_fbs
