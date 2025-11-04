import torch
try:
    import torch_sdaa
except:
    print("import torch_sdaa failed, if you want to use sdaa, please install it first")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from time import time

# 初始化logger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips", {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 启动分布式训练DDP时，从外部获取LOCAL_RANK 和WORLD_SIZE参数
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["LOCAL_RANK"])

# 设置模型运行设备，并且初始化通信进程组
def setup_device_and_ddp(device, rank):
    if device == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl")
        return torch.device("cuda", rank)
    elif device == "sdaa":
        torch.sdaa.set_device(rank)
        dist.init_process_group("tccl")
        return torch.sdaa.current_device()
    else:
        raise ValueError(f"Invalid device: {device}, please choose cuda or sdaa")

def main(device, epochs, batch_size, lr):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = setup_device_and_ddp(device, rank)

    model = MLP().to(device)
    ddp_model = DDP(model, device_ids=[rank])  # 构建DDP模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr)
    scaler = GradScaler()

    # 训练模型
    for epoch in range(epochs):
        ddp_model.train()
        running_loss = 0.0
        _time = time()  # 初始化时间
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.to(memory_format=torch.channels_last)
            optimizer.zero_grad()

            with autocast():
                output = ddp_model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            ips = batch_size / (time() - _time)  # 计算每秒处理的样本数
            _time = time()  # 更新当前时间戳

            # 记录日志
            json_logger.log(
                step=(epoch, batch_idx),
                data={
                    "rank": rank,
                    "train.loss": loss.item(),
                    "train.ips": ips,
                },
                verbosity=Verbosity.DEFAULT,
            )

            if batch_idx % 100 == 99:  # 每100个batch打印一次
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 100:.6f}')
                running_loss = 0.0

        # 测试模型
        ddp_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.to(memory_format=torch.channels_last)
                with autocast():
                    output = ddp_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch: {epoch + 1}, Accuracy: {accuracy:.2f}%')

        # 记录测试精度日志
        json_logger.log(
            step=(epoch, "test"),
            data={
                "rank": rank,
                "test.accuracy": accuracy,
            },
            verbosity=Verbosity.DEFAULT,
        )

        if accuracy > 95:
            print("Training stopped as accuracy reached 95%.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用TecoPyTorch进行分布式训练')
    parser.add_argument('--device', type=str, required=True, help='设备类型，cuda或sdaa')
    parser.add_argument('--epochs', type=int, help='训练模型的总epoch数')
    parser.add_argument('--batch_size', type=int, help='训练时的批量大小')
    parser.add_argument('--lr', type=float, help='优化器的学习率')
    args = parser.parse_args()

    main(args.device, args.epochs, args.batch_size, args.lr)
