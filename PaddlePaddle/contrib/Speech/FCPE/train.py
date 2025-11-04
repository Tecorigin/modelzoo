#encoding=utf-8
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
import paddlefcpe
import os
os.environ["CUSTOM_DEVICE_BLACK_LIST"] = 'concat,conv2d,conv2d_grad'
import argparse
import paddle
from paddle.optimizer import lr as lr_scheduler
import paddle.distributed as dist
from savertools import utils
from data_loaders_wav import get_data_loaders
from solver_wav import train
import paddlefcpe

try:
    import paddle_sdaa # 某国产卡
except ImportError:
    pass

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    # parse commands
    cmd = parse_args()

    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

    # device
    place = paddle.CPUPlace()
    if args.device in ['cuda', 'gpu']:
        place = paddle.set_device("gpu")
    if args.device == 'sdaa':
        place = paddle.CustomPlace("sdaa",0)

    dist.init_parallel_env() # DDP init
    # load model
    model = paddlefcpe.spawn_model(args)
    model.to(args.device)
    model = paddle.DataParallel(model) # DDP model
    # load parameters
    scheduler = lr_scheduler.StepDecay(0.001, args.train.decay_step, gamma=args.train.gamma, last_epoch=-1, verbose=False)
    optimizer = paddle.optimizer.AdamW(learning_rate=scheduler, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=model.parameters(), weight_decay=0.01, lr_ratio=None, apply_decay_param_fun=None, grad_clip=None, lazy_mode=False, multi_precision=False, name=None)
    
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    scheduler.last_epoch = initial_global_step - 2
    scheduler.last_epoch += 1
    
    scheduler.base_lr = args.train.lr
    scheduler.last_lr = (args.train.lr * args.train.gamma ** max((initial_global_step - 2) // args.train.decay_step,0))
    optimizer._weight_decay=args.train.weight_decay

    # datas
    loader_train, loader_valid = get_data_loaders(args)

    # run
    train(args, initial_global_step, model, optimizer, scheduler, loader_train, loader_valid)
