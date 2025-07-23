import argparse

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train DenseNet on sdaa')
    parser.add_argument('--data_path', type=str, default='/data/teco-data/imagenet', help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='(default is 300) Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Model save path')
    parser.add_argument('--num_steps', type=int, default=100, help='(default is 3100)Number of training steps')
    # 添加 DenseNet 所需的参数
    parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet')
    parser.add_argument('--dropRate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--reduction', type=float, default=0.5, help='Reduction rate at transition layers')
    parser.add_argument('--bottleneck', action='store_true', help='Use bottleneck structure')
    parser.add_argument('--depth', type=int, default=121, help='Depth of DenseNet (e.g., 121, 169, 201)')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'], help='Dataset to use')
    parser.add_argument('--optMemory', type=int, default=2, help='Memory optimization level')
    parser.add_argument('--cudnn', type=str, default='deterministic', choices=['deterministic', 'default'], help='cuDNN mode')
    return parser.parse_args()