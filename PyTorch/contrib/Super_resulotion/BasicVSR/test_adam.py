import torch 
import torch_sdaa

from torch_sdaa.optim.optimizer import Adam

import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import faulthandler
faulthandler.enable()

torch.manual_seed(42)

# 创建一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 生成一些随机数据
num_samples = 1000
input_size = 10
X = torch.randn(num_samples, input_size)
y = torch.randn(num_samples, 1)

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = "sdaa:0"

# 初始化模型、损失函数和优化器
model = SimpleNet()
model.to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

model.train()
# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in dataloader:
        # 前向传播
        batch_X,batch_y = batch_X.to(device),batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

