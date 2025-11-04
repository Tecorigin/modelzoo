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
from os import getcwd

from utils.utils import get_classes

# -------------------------------------------------------------------#
#   classes_path    指向model_data下的txt，与自己训练的数据集相关 
#                   训练前一定要修改classes_path，使其对应自己的数据集
#                   txt文件中是自己所要去区分的种类
#                   与训练和预测所用的classes_path一致即可
# -------------------------------------------------------------------#
classes_path = 'model_data/cls_classes.txt'
# -------------------------------------------------------#
#   datasets_path   指向数据集所在的路径
# -------------------------------------------------------#
datasets_path = 'datasets'

sets = ["train", "test"]
classes, _ = get_classes(classes_path)

if __name__ == "__main__":
    for se in sets:
        list_file_path = os.path.join(datasets_path, 'cls_' + se + '.txt')
        list_file = open(list_file_path, 'w')
        # list_file = open('cls_' + se + '.txt', 'w')

        datasets_path_t = os.path.join(datasets_path, se)
        types_name = os.listdir(datasets_path_t)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)

            photos_path = os.path.join(datasets_path_t, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s' % (os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()
