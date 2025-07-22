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
import argparse
import ast
import sys
import warnings


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            if '=' not in value:
                raise argparse.ArgumentError(self, f"Expected key=value format but got '{value}'")
            key, value = value.split('=', 1)
            try:
                kw[key] = ast.literal_eval(value)
            except Exception:
                kw[key] = value  # fallback to string
        setattr(namespace, self.dest, kw)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="训练启动参数解析")

    # 基础数据参数
    parser.add_argument("--train-data", type=str, default=None,
                        help="训练数据路径，支持 webdataset 多源用 :: 分隔")
    parser.add_argument("--train-data-upsampling-factors", type=str, default=None,
                        help="多数据源采样比例，格式如 '1::2::0.5'")
    parser.add_argument("--val-data", type=str, default=None, help="验证数据路径")
    parser.add_argument("--train-num-samples", type=int, default=None,
                        help="训练集样本数，webdataset 无 info 文件时需指定")
    parser.add_argument("--val-num-samples", type=int, default=None,
                        help="验证集样本数，webdataset 无 info 文件时需指定")
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "synthetic", "auto"],
                        default="auto", help="数据集类型")
    parser.add_argument("--dataset-resampled", action="store_true", default=False,
                        help="是否对 webdataset 进行替换采样")
    parser.add_argument("--csv-separator", type=str, default=",",
                        help="csv 格式数据分隔符")
    parser.add_argument("--csv-img-key", type=str, default="filepath",
                        help="csv 格式中图片路径字段名")
    parser.add_argument("--csv-caption-key", type=str, default="title",
                        help="csv 格式中文本字段名")
    parser.add_argument("--imagenet-val", type=str, default=None,
                        help="ImageNet 验证集路径（零样本评估）")
    parser.add_argument("--imagenet-v2", type=str, default=None,
                        help="ImageNet V2 路径（零样本评估）")

    # 日志和缓存
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="模型和分词器缓存目录")
    parser.add_argument("--logs", type=str, default="./logs/",
                        help="tensorboard 日志保存目录，设 None 不保存")
    parser.add_argument("--log-local", action="store_true", default=False,
                        help="是否只在本地 master 记录日志")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称，默认用时间戳")

    # 训练参数
    parser.add_argument("--workers", type=int, default=4, help="dataloader 线程数/每GPU")
    parser.add_argument("--batch-size", type=int, default=64, help="每GPU批大小")
    parser.add_argument("--epochs", type=int, default=32, help="训练轮数")
    parser.add_argument("--epochs-cooldown", type=int, default=None,
                        help="学习率冷却轮数，从末尾开始计数")

    # 优化器参数
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon")
    parser.add_argument("--wd", type=float, default=0.2, help="权重衰减")
    parser.add_argument("--momentum", type=float, default=None, help="动量（部分优化器用）")
    parser.add_argument("--warmup", type=int, default=10000, help="warmup 步数")
    parser.add_argument("--opt", type=str, default='adamw',
                        help="优化器，adamw 或 timm/xxx")

    # 训练策略
    parser.add_argument("--use-bn-sync", action="store_true", default=False, help="同步 BatchNorm")
    parser.add_argument("--skip-scheduler", action="store_true", default=False, help="跳过学习率衰减")
    parser.add_argument("--lr-scheduler", type=str, default='cosine',
                        help="学习率策略: cosine, const, const-cooldown")
    parser.add_argument("--lr-cooldown-end", type=float, default=0.0, help="冷却末端学习率")
    parser.add_argument("--lr-cooldown-power", type=float, default=1.0, help="冷却功率")

    # 保存与验证
    parser.add_argument("--save-frequency", type=int, default=1, help="多少轮保存模型")
    parser.add_argument("--save-most-recent", action="store_true", default=False,
                        help="是否始终保存最近一次模型到 epoch_latest.pt")
    parser.add_argument("--zeroshot-frequency", type=int, default=2, help="多少轮做零样本评估")
    parser.add_argument("--val-frequency", type=int, default=1, help="多少轮做验证评估")
    parser.add_argument("--resume", type=str, default=None, help="断点续训模型路径")

    # 精度相关
    parser.add_argument("--precision", choices=["amp", "amp_bf16", "amp_bfloat16", "bf16",
                                                "fp16", "pure_bf16", "pure_fp16", "fp32"],
                        default="amp", help="计算精度")

    # 模型相关
    parser.add_argument("--model", type=str, default="RN50", help="模型名")
    parser.add_argument("--pretrained", type=str, default='', help="预训练权重路径或tag")
    parser.add_argument("--pretrained-image", action="store_true", default=False,
                        help="加载imagenet预训练权重")
    parser.add_argument("--lock-image", action="store_true", default=False, help="锁定图像塔梯度")
    parser.add_argument("--lock-image-unlocked-groups", type=int, default=0,
                        help="图像塔最后解锁层数")
    parser.add_argument("--lock-image-freeze-bn-stats", action="store_true", default=False,
                        help="冻结图像塔BatchNorm统计")

    # 图像处理
    parser.add_argument('--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='覆盖图像均值')
    parser.add_argument('--image-std', type=float, nargs='+', default=None, metavar='STD',
                        help='覆盖图像标准差')
    parser.add_argument('--image-interpolation', type=str, choices=['bicubic', 'bilinear', 'random'],
                        default=None, help="图像插值方式")
    parser.add_argument('--image-resize-mode', type=str, choices=['shortest', 'longest', 'squash'],
                        default=None, help="图像resize模式")

    # 数据增强配置字典
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs,
                        help="增强配置，格式 key=value")

    # 其他训练选项
    parser.add_argument("--grad-checkpointing", action="store_true", default=False, help="梯度检查点")
    parser.add_argument("--local-loss", action="store_true", default=False, help="局部loss计算")
    parser.add_argument("--gather-with-grad", action="store_true", default=False,
                        help="特征聚合带梯度")

    # 强制覆盖配置
    parser.add_argument("--force_context_length", type=int, default=None, help="上下文长度覆盖")
    parser.add_argument("--force-image-size", type=int, nargs='+', default=None, help="图像尺寸覆盖")
    parser.add_argument("--force-quick-gelu", action="store_true", default=False,
                        help="强制QuickGELU激活")
    parser.add_argument("--force-patch-dropout", type=float, default=None, help="patch dropout覆盖")
    parser.add_argument("--force-custom-text", action="store_true", default=False,
                        help="使用自定义文本模型")

    # torch脚本相关
    parser.add_argument("--torchscript", action="store_true", default=False, help="torch.jit.script")
    parser.add_argument("--torchcompile", action="store_true", default=False, help="torch.compile")
    parser.add_argument("--trace", action="store_true", default=False, help="torch.jit.trace")

    # 训练细节
    parser.add_argument("--accum-freq", type=int, default=1, help="梯度累计步数")
    parser.add_argument("--device", type=str, default="sdaa", help="设备")

    # 分布式相关
    parser.add_argument("--dist-url", type=str, default=None, help="分布式初始化url")
    parser.add_argument("--dist-backend", type=str, default=None,
                        help="分布式后端，nccl或hccl")
    parser.add_argument("--report-to", type=str, default='',
                        help="日志报告平台 wandb, tensorboard")
    parser.add_argument("--wandb-notes", type=str, default='', help="wandb备注")
    parser.add_argument("--wandb-project-name", type=str, default='open-clip', help="wandb项目名")

    # 其他标志
    parser.add_argument("--debug", action="store_true", default=False, help="调试信息")
    parser.add_argument("--copy-codebase", action="store_true", default=False, help="复制代码库")
    parser.add_argument("--horovod", action="store_true", default=False, help="使用Horovod")
    parser.add_argument("--ddp-static-graph", action="store_true", default=False,
                        help="DDP静态图优化")
    parser.add_argument("--no-set-device-rank", action="store_true", default=False,
                        help="不根据local rank设置设备索引")

    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="梯度裁剪阈值")

    # 文本锁定相关
    parser.add_argument("--lock-text", action="store_true", default=False, help="锁定文本塔梯度")
    parser.add_argument("--lock-text-unlocked-layers", type=int, default=0,
                        help="文本塔解锁层数")
    parser.add_argument("--lock-text-freeze-layer-norm", action="store_true", default=False,
                        help="冻结文本塔LayerNorm统计")

    parser.add_argument("--log-every-n-steps", type=int, default=100, help="日志打印步数")

    # CoCa相关loss权重
    parser.add_argument("--coca-caption-loss-weight", type=float, default=2.0,
                        help="CoCa Caption loss 权重")
    parser.add_argument("--coca-contrastive-loss-weight", type=float, default=1.0,
                        help="CoCa Contrastive loss 权重")

    # 远程同步相关
    parser.add_argument("--remote-sync", type=str, default=None, help="远程同步路径")
    parser.add_argument("--remote-sync-frequency", type=int, default=300,
                        help="远程同步频率（秒）")
    parser.add_argument("--remote-sync-protocol", choices=["s3", "fsspec"], default="s3",
                        help="远程同步协议")

    parser.add_argument("--delete-previous-checkpoint", action="store_true", default=False,
                        help="保存新checkpoint时删除旧checkpoint")

    # 蒸馏相关
    parser.add_argument("--distill-model", type=str, default=None, help="蒸馏模型架构")
    parser.add_argument("--distill-pretrained", type=str, default=None, help="蒸馏权重")

    # bitsandbytes相关
    parser.add_argument("--use-bnb-linear", type=str, default=None,
                        help="bitsandbytes线性层替换")

    # 其他
    parser.add_argument("--siglip", action="store_true", default=False, help="使用SigLip损失")
    parser.add_argument("--loss-dist-impl", type=str, default=None, help="分布式loss实现")

    args = parser.parse_args(args)

    # 校验简单示例
    if args.batch_size <= 0:
        warnings.warn("batch-size 应为正整数")
        sys.exit(1)
    if args.epochs <= 0:
        warnings.warn("epochs 应为正整数")
        sys.exit(1)
    if args.lr is not None and args.lr <= 0:
        warnings.warn("lr 应为正数")
        sys.exit(1)

    # 根据模型名设置默认优化器参数（如果opt不是timm系列）
    if 'timm' not in args.opt:
        default_params = get_default_params(args.model)
        for key, val in default_params.items():
            if getattr(args, key) is None:
                setattr(args, key, val)

    return args


if __name__ == "__main__":
    parsed = parse_args()
    print("Parsed arguments:")
    for k, v in vars(parsed).items():
        print(f"  {k}: {v}")
