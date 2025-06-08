import argparse
import os
from mmengine.config import DictAction

def parse_args():
    parser = argparse.ArgumentParser(description='Unified training interface for SAE models')

    # 基本训练参数
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--work-dir', type=str, default=None, help='Directory to save logs and checkpoints')
    parser.add_argument('--model-name', type=str, default='CAE', help='Name of the model')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300, help='Total number of training epochs')
    parser.add_argument('--dataset-root', type=str, default='/data/teco-data/imagenet/', help='Root path to dataset')

    # Mixed precision 和 恢复训练相关
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--resume', nargs='?', type=str, const='auto', help='Checkpoint path or auto to resume latest')
    parser.add_argument('--no-validate', action='store_true', help='Disable validation during training')
    parser.add_argument('--auto-scale-lr', action='store_true', help='Auto-scale learning rate based on batch size')

    # dataloader 优化选项
    parser.add_argument('--no-pin-memory', action='store_true', help='Disable pin_memory in dataloaders')
    parser.add_argument('--no-persistent-workers', action='store_true', help='Disable persistent workers in dataloaders')

    # mmengine 相关
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the config file, in key=value format')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='Job launcher type')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args