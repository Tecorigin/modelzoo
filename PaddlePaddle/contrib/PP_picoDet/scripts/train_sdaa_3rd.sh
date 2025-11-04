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

#!/bin/bash
cd ..
# 运行后面的命令
pip uninstall -y numpy
pip install -r requirements.txt

echo "开始运行训练命令"
# training on multi-GPU: picodet
export PADDLE_XCCL_BACKEND=sdaa
export PADDLE_DISTRI_BACKEND=tccl
export SDAA_VISIBLE_DEVICES=0,1,2,3
# export SDAA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/picodet/picodet_xs_320_coco_lcnet.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log
# python -m paddle.distributed.launch --gpus 0 tools/train.py -c configs/picodet/picodet_xs_320_coco_lcnet.yml --eval --amp 2>&1|tee scripts/train_sdaa_3rd.log


# 检查命令是否执行成功
if [ $? -ne 0 ]; then
    echo "训练命令执行失败"
    exit 1
fi


echo "脚本执行完毕"