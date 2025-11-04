# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted to tecorigin hardware

import dllogger
from image_classification.optimizer import (
    get_optimizer,
    create_optimizer_fused_resnet50_lars,
    create_optimizer_lars,
)
from image_classification.lr import (
    lr_cosine_policy,
    lr_linear_policy,
    lr_step_policy,
    no_lr_policy,
    MLPerfLRScheduler,
)
from image_classification.models import (
    resnet18,
    resnet101,
    resnet50,
    resnext101_32x4d,
    se_resnext101_32x4d,
    resnet50_v10,
)

try:
    from image_classification.models import (
        fused_resnet50)
    USE_FUSED = True
except:
    USE_FUSED = False
    print('the torch_sdaa not support models,please use the newest ')
from image_classification.utils.utils import *
from image_classification.train.training import *
from image_classification.data.dataloaders import *
from image_classification.utils.mixup import NLLMultiLabelSmooth, MixUpWrapper
from image_classification.utils.smoothing import LabelSmoothing
import image_classification.utils.logger as log
import torch.utils.data.distributed
import torch.utils.data
import torch.optim
import torch.nn.parallel
import torch.distributed as dist
from torchvision import models
from copy import deepcopy
import random
import argparse
import os
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
from image_classification.utils.tcap_logger import get_tcap_logger

os.environ[
    "KMP_AFFINITY"
] = "disabled"  # We need to do this before importing anything else as a workaround for this bug: https://github.com/pytorch/pytorch/issues/28389

try:
    import torch_sdaa
    import torch_sdaa.core.sdaa_model as sm
    import torch_sdaa.distributed as sdaa_dist
except:
    print('import torch_sdaa failed')

def str2bool(v):
    """
    将命令行输入的str转换为布尔值
    :param v: str值
    :return:
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def available_models():
    models = {
        m.name: m
        for m in [
            resnet18,
            resnet101,
            resnet50,
            resnext101_32x4d,
            se_resnext101_32x4d,
            resnet50_v10,
        ]
    }
    if USE_FUSED :
        models.update({fused_resnet50.name:fused_resnet50})
    return models


def add_parser_arguments(parser, skip_arch=False):
    parser.add_argument("--data", type=str, metavar="DIR",
                        help="path to dataset")
    parser.add_argument(
        "--data-backend",
        metavar="BACKEND",
        default="dali-cpu",
        choices=DATA_BACKEND_CHOICES,
        help="data backend: "
        + " | ".join(DATA_BACKEND_CHOICES)
        + " (default: dali-cpu)",
    )
    parser.add_argument(
        "--interpolation",
        metavar="INTERPOLATION",
        default="bilinear",
        help="interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)",
    )
    if not skip_arch:
        model_names = available_models().keys()
        parser.add_argument(
            "--arch",
            "-a",
            metavar="ARCH",
            default="resnet50",
            help="model architecture: "
            + " | ".join(model_names)
            + " (default: resnet50)",
        )

    parser.add_argument(
        "-j",
        "--workers",
        default=5,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 5)",
    )
    parser.add_argument(
        "--prefetch",
        default=2,
        type=int,
        metavar="N",
        help="number of samples prefetched by each loader",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--run-epochs",
        default=-1,
        type=int,
        metavar="N",
        help="run only N epochs, used for checkpointing runs",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=-1,
        type=int,
        metavar="N",
        help="early stopping after N epochs without validation accuracy improving",
    )
    parser.add_argument(
        "--image-size", default=None, type=int, help="resolution of image"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per gpu",
    )

    parser.add_argument(
        "--optimizer-batch-size",
        default=-1,
        type=int,
        metavar="N",
        help="size of a total batch size, for simulating bigger batches using gradient accumulation",
    )

    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--lr-schedule",
        default="step",
        type=str,
        metavar="SCHEDULE",
        choices=["step", "linear", "cosine"],
        help="Type of LR schedule: {}, {}, {}".format(
            "step", "linear", "cosine"),
    )

    parser.add_argument("--end-lr", default=0, type=float)

    parser.add_argument(
        "--warmup", default=0, type=int, metavar="E", help="number of warmup epochs"
    )

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        metavar="S",
        help="label smoothing",
    )
    parser.add_argument(
        "--mixup", default=0.0, type=float, metavar="ALPHA", help="mixup alpha"
    )
    parser.add_argument(
        "--optimizer", default="sgd", type=str, choices=("sgd", "rmsprop", "adam")
    )

    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--bn-weight-decay",
        action="store_true",
        help="use weight_decay on batch normalization learnable parameters, (default: false)",
    )
    parser.add_argument(
        "--rmsprop-alpha",
        default=0.9,
        type=float,
        help="value of alpha parameter in rmsprop optimizer (default: 0.9)",
    )
    parser.add_argument(
        "--rmsprop-eps",
        default=1e-3,
        type=float,
        help="value of eps parameter in rmsprop optimizer (default: 1e-3)",
    )

    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="use nesterov momentum, (default: false)",
    )

    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--static-loss-scale",
        type=float,
        default=1,
        help="Static loss scale, positive power of 2 values can improve amp convergence.",
    )
    parser.add_argument(
        "--dynamic-loss-scale",
        action="store_true",
        help="Use dynamic loss scaling.  If supplied, this argument supersedes "
        + "--static-loss-scale.",
    )
    parser.add_argument(
        "--prof", type=int, default=-1, metavar="N", help="Run only N iterations"
    )
    parser.add_argument(
        "--amp",
        type=str2bool, default=False,
        help="Run model AMP (automatic mixed precision) mode.",
    )
    parser.add_argument(
        "--compile",
        type=str2bool, default=False,
        help="enable compilation.",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="random seed used for numpy and pytorch"
    )

    parser.add_argument(
        "--gather-checkpoints",
        default="0",
        type=int,
        help=(
            "Gather N last checkpoints throughout the training,"
            " without this flag only best and last checkpoints will be stored. "
            "Use -1 for all checkpoints"
        ),
    )

    parser.add_argument(
        "--raport-file",
        default="experiment_raport.json",
        type=str,
        help="file in which to store JSON experiment raport",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate checkpoint/model"
    )
    parser.add_argument("--training-only",
                        type=str2bool,
                        default=False,
                        help="do not evaluate")

    parser.add_argument(
        "--no-checkpoints",
        type=str2bool,
        default=False,
        dest="save_checkpoints",
        help="do not store any checkpoints, useful for benchmarking",
    )
    parser.add_argument(
        "--jit",
        type=str,
        default="no",
        choices=["no", "script"],
        help="no -> do not use torch.jit; script -> use torch.jit.script",
    )

    parser.add_argument("--checkpoint-filename",
                        default="checkpoint.pth.tar", type=str)

    parser.add_argument(
        "--workspace",
        type=str,
        default="./",
        metavar="DIR",
        help="path to directory where checkpoints will be stored",
    )
    parser.add_argument(
        "--memory-format",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="memory layout, nchw or nhwc",
    )
    parser.add_argument("--use-ema", default=None, type=float, help="use EMA")
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        choices=[None, "autoaugment"],
        help="augmentation method",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        required=False,
    )

    parser.add_argument(
        '--grad_scaler',
        type=str2bool,
        default=False,
        help='whether to use GradScaler')

    parser.add_argument(
        '--PrefetchedWrapper',
        action='store_true',
        default=False,
        help='whether to use PrefetchedWrapper function'
    )

    parser.add_argument(
        '--pin_memory',
        action='store_true',
        default=False,
        help='pin_memory function'
    )

    parser.add_argument(
        '--worker_init_fn',
        action='store_true',
        default=False,
        help='worker_init_fn function'
    )

    parser.add_argument(
        '--collate_fn',
        action='store_true',
        default=False,
        help='dataset collate_fn'
    )

    parser.add_argument(
        '--drop_last',
        action='store_true',
        default=False,
        help='dataset drop_last'
    )

    parser.add_argument(
        '--persistent_workers',
        action='store_true',
        default=False,
        help='dataset persistent_workers'
    )

    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help='which device to use. cpu, cuda, sdaa optional, cpu default'
    )

    parser.add_argument(
        '--profiler',
        type=str2bool,
        default=False,
        help='use profiler function'
    )

    parser.add_argument(
        '--tensorboard',
        action='store_true',
        default=False,
        help='use tensorboard'
    )

    parser.add_argument(
        '--layer_diff',
        type=str2bool,
        default=False,
        help='use layer diff check tool'
    )

    parser.add_argument("--check_dir", type=str,
                        default="./", help="dir to save check pt")

    parser.add_argument(
        '--channel_last',
        action='store_true',
        default=False,
        help='set dats to channel_last data format'
    )

    parser.add_argument(
        '--FP64',
        action='store_true',
        default=False,
        help='fp64 train when use cpu'
    )

    parser.add_argument(
        '--model_cl',
        action='store_true',
        default=False,
        help='set model to channel_last'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
    )

    parser.add_argument(
        '--profiler_path',
        type=str,
        default=None,
        help='path to save profiler result'
    )

    parser.add_argument(
        '--fused_optimizer',
        type=str2bool,
        default=False,
        help='use fused optimizer'
    )

    parser.add_argument(
        '--use_stream',
        action='store_true',
        default=False,
        help='use data stream'
    )

    parser.add_argument(
        '--early_stop',
        type=int,
        default=-1,
        help='early stop when training'
    )

    parser.add_argument(
        '--mlperf_mode',
        type=str2bool,
        default=False,
        help='use mlperf mode '
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet',
        help='which dataset to use '
    )

def prepare_for_training(args, model_args, model_arch):
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        print('world_size', int(os.environ["WORLD_SIZE"]))
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.local_rank = 0
        args.dev = 0
        args.world_size = 1

    device = torch.device(type=args.device, index=0)

    if args.distributed:
        device = prepare_for_distributed(args)

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            # Worker process should inherit its affinity from parent
            affinity = os.sched_getaffinity(0)
            print(
                f"Process {args.local_rank} Worker {id} set affinity to: {affinity}")

            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:
        def _worker_init_fn(id):
            # Worker process should inherit its affinity from parent
            affinity = os.sched_getaffinity(0)
            print(
                f"Process {args.local_rank} Worker {id} set affinity to: {affinity}")

    if args.static_loss_scale != 1.0:
        if not args.amp:
            print("Warning: if --amp is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size
        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cpu()
            )
            start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            if "state_dict_ema" in checkpoint:
                model_state_ema = checkpoint["state_dict_ema"]
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if start_epoch >= args.epochs:
                print(
                    f"Launched training for {args.epochs}, checkpoint already run {start_epoch}"
                )
                exit(1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            model_state_ema = None
            optimizer_state = None
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None

    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        def loss(): return NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        def loss(): return LabelSmoothing(args.label_smoothing)

    memory_format = (
        torch.channels_last if args.channel_last else torch.contiguous_format
    )

    # Create data loaders and optimizers as needed
    if args.data_backend == "pytorch":
        if args.dataset == 'imagenet':
            get_train_loader = get_pytorch_train_loader
            get_val_loader = get_pytorch_val_loader
        elif args.dataset == 'cifar10':
            get_train_loader = get_cifar10_train_loader 
            get_val_loader =   get_cifar10_val_loader 
            model_args.num_classes = 10
        elif args.dataset == 'flowers':
            model_args.num_classes = 102
            get_train_loader = get_pytorch_train_loader
            get_val_loader = get_pytorch_val_loader
    elif args.data_backend == "synthetic":
        get_val_loader = get_synthetic_loader
        get_train_loader = get_synthetic_loader
    else:
        print("Bad databackend picked")
        exit(1)
        
    model = model_arch(
        **{
            k: v
            if k != "pretrained"
            else v and (not args.distributed or dist.get_rank() == 0)
            for k, v in model_args.__dict__.items()
        }
    )

    image_size = (
        args.image_size
        if args.image_size is not None
        else model.arch.default_image_size
    )

    if args.grad_scaler:
        GradScaler = get_grad_scaler(args)
        scaler = GradScaler(
            init_scale=args.static_loss_scale,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=100 if args.dynamic_loss_scale else 2000,
            enabled=args.grad_scaler,
        )
    else:
        scaler = None

    # tensorboard
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_path = os.path.join(f"./tb_log/{args.time}", str(device))
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        writer = SummaryWriter(log_dir=tb_path)
    else:
        writer = None


    executor = Executor(
        args,
        device,
        model,
        loss(),
        memory_format=memory_format,
        amp=args.amp,
        scaler=scaler,
        divide_loss=batch_size_multiplier,
        ts_script=args.jit == "script",
    )

    

    train_loader, train_loader_len = get_train_loader(
        args,
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        args.mixup > 0.0,
        interpolation=args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, val_loader_len = get_val_loader(
        args,
        args.data,
        image_size,
        args.batch_size,
        model_args.num_classes,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        _worker_init_fn=_worker_init_fn,
        memory_format=memory_format,
        prefetch_factor=args.prefetch,
    )

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if not os.path.exists(args.workspace):
            os.mkdir(args.workspace)
        logger_res = get_tcap_logger(args)
    else:
        logger_res = get_tcap_logger(args)

    if args.mlperf_mode:
        if 'fuse' in args.arch:
            optimizer = create_optimizer_fused_resnet50_lars(model=model, lr=args.lr, epsilon=0,
                                            momentum=0.9, weight_decay=5.0e-05,
                                            bn_bias_separately=True)
        else:
            optimizer = create_optimizer_lars(model=model, lr=args.lr, epsilon=0,
                                        momentum=0.9, weight_decay=5.0e-05,
                                        bn_bias_separately=True)
        num_steps_per_epoch = len(train_loader)
        lr_policy = MLPerfLRScheduler(optimizer, args.epochs, args.warmup,
                                                        num_steps_per_epoch, args.lr,start_epoch,
                                                        0.0001, 2.0)
    else:
        optimizer = get_optimizer(
            list(executor.model.named_parameters()),
            args.lr,
            args=args,
            state=optimizer_state,
        )

        if args.lr_schedule == "step":
            lr_policy = lr_step_policy(args.lr, [30, 60, 80], 0.1, args.warmup)
        elif args.lr_schedule == "cosine":
            lr_policy = lr_cosine_policy(
                args.lr, args.warmup, args.epochs, end_lr=args.end_lr
            )
        elif args.lr_schedule == "linear":
            lr_policy = lr_linear_policy(args.lr, args.warmup, args.epochs)
        elif not args.lr_schedule:
            lr_policy = no_lr_policy(args.lr)


    if args.distributed:
        executor.distributed(args.device, args.dev)

    if args.compile:
        backend = 'teco_inductor' if 'sdaa' in args.device else 'inductor'
        executor.model = torch.compile(executor.model, backend=backend)

    if model_state is not None:
        executor.model.load_state_dict(model_state)

    trainer = Trainer(
        executor,
        optimizer,
        grad_acc_steps=batch_size_multiplier,
        ema=args.use_ema,
        writer=writer,
    )

    if (args.use_ema is not None) and (model_state_ema is not None):
        trainer.ema_executor.model.load_state_dict(model_state_ema)

    return (
        model,
        device,
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger_res,
        start_epoch,
        writer,
    )

def get_grad_scaler(args):
    if args.device == 'sdaa':
        from torch_sdaa.amp import GradScaler
    else:
        from torch.cuda.amp import GradScaler
    return GradScaler


def prepare_for_distributed(args):
    if args.device == 'sdaa':
        args.dev = args.local_rank % sm.device_count()
        sm.set_device(args.dev)
        dist.init_process_group("tccl")
        args.world_size = torch.distributed.get_world_size()
        device = sm.sdaa_device(args.dev)
    elif args.device == 'cuda':
        args.dev = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.dev)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
        device = torch.device(args.dev)
    else:
        print('CPU does not support distributed training')
        raise SystemError
    return device

def main(args, model_args, model_arch):

    global best_prec1
    best_prec1 = 0

    (
        model,
        device,
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger_res,
        start_epoch,
        writer,
    ) = prepare_for_training(args, model_args, model_arch)


    train_loop(
        args,
        device,
        trainer,
        lr_policy,
        train_loader,
        train_loader_len,
        val_loader,
        logger_res,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs)
        if args.run_epochs != -1
        else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        prof=args.prof,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.workspace,
        checkpoint_filename=args.checkpoint_filename,
        keep_last_n_checkpoints=args.gather_checkpoints,
        topk=args.topk,
        writer=writer,
    )

    print("Experiment ended")


if __name__ == "__main__":

    pass
