import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(args):
    """加载和预处理数据集"""
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        if args.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
            val_dataset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=val_transform)
        else:  # cifar100
            train_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
            val_dataset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=val_transform)
    else:  # imagenet
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = torchvision.datasets.ImageNet(root=args.data_path, split='train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageNet(root=args.data_path, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    return train_loader, val_loader