import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Unified training interface for FER model")
    parser.add_argument('--model_name', type=str, default='VGG19', help='Model name, e.g., VGG19 or Resnet18')
    parser.add_argument('--batchsize', type=int, default=128, help='Training batch size')
    parser.add_argument('--epoch', type=int, default=250, help='Total number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    return parser.parse_args()