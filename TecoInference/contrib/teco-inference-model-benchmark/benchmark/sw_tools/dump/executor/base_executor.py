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

import abc


class BaseExecutor(abc.ABC):
    """
    usage:
        ...
        executor = XXXExecutor()
        ...
        executor.init(init_configs)
        ...
        executor.load_model(model, load_model_configs)
        ...
        outputs = executor.execute(output_names, inputs, execute_configs)
        ...
        executor.release()
        ...
    """

    @abc.abstractmethod
    def init(self, init_configs=None):
        """
        init_configs:
            configurations for initialize
        """

    @abc.abstractmethod
    def load_model(self, model, load_model_configs=None):
        """
        model:
            onnx model
        load_model_configs:
            configurations for model
        """

    @abc.abstractmethod
    def execute(self, output_names, inputs, execute_configs=None):
        """
        return:
            output numpy array list by output_names

        output_names:
            a list of output tensor names in onnx model
        inputs:
            a map from input tensor name to input array
        execute_configs:
            configurations for execution
        """

    @abc.abstractmethod
    def release(self):
        pass
