# argument.py
#
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # 模型与路径配置
    parser.add_argument('--model_name_or_path', type=str, default='model',
                        help='本地模型路径或模型名称，如 google/electra-small-discriminator')
    parser.add_argument('--output_dir', type=str, default='./electra_out',
                        help='模型输出目录')
    parser.add_argument('--config_name', type=str, default=None, help='可选，config 路径')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='可选，tokenizer 路径')

    # 数据配置
    parser.add_argument("--train_file", type=str, default="../configs/train_sample.txt")
    parser.add_argument('--validation_file', type=str, default='../configs/train_sample.txt',
                        help='验证数据文件')
    parser.add_argument('--max_seq_length', type=int, default=32)

    # 模型任务设置
    parser.add_argument('--line_by_line', action='store_true')
    parser.add_argument('--mlm_probability', type=float, default=0.15)

    # 训练参数
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--save_steps', type=int, default=150)
    parser.add_argument("--logging_steps", type=int, default=1,
                    help="每隔多少步打印一次日志")


    # 启动设置
    parser.add_argument('--local_rank', type=int, default=-1, help='用于分布式训练')

    # AMP 设置
    parser.add_argument('--fp16', action='store_true', help='启用混合精度训练')

    return parser.parse_args()
