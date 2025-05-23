# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted to tecorigin hardware

import sys
import inspect

from .legendary_models.resnet import ResNet18, ResNet18_vd, ResNet34, ResNet34_vd, ResNet50, ResNet50_vd, ResNet101, ResNet101_vd, ResNet152, ResNet152_vd, ResNet200_vd


# help whl get all the models' api (class type) and components' api (func type)
def get_apis():
    current_func = sys._getframe().f_code.co_name
    current_module = sys.modules[__name__]
    api = []
    for _, obj in inspect.getmembers(current_module,
                                     inspect.isclass) + inspect.getmembers(
                                         current_module, inspect.isfunction):
        api.append(obj.__name__)
    api.remove(current_func)
    return api


__all__ = get_apis()
