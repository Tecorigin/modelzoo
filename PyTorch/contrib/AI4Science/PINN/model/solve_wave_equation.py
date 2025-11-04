# coding=utf-8

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

import torch
import torch_sdaa
from torch_sdaa.amp import GradScaler, autocast
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import os
from torch.nn.parallel import DistributedDataParallel

# solve wave equation
#   utt(x, t) = uxx(x, t)
#   u(x, 0) = cos(x)
#   ut(x, 0) = -sin(x)
# where x in [0, torch.pi / 2]，t in [0, torch.pi / 2]
# its analytical solution is u(x, t) = cos(x + t)


# construct neural network (MLP model)
class PhysicsInformedNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(PhysicsInformedNeuralNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)

# control equation, border condition and sampling

# utt(x, t) - uxx(x, t) = 0
def control(n, device):
    x = torch.rand(size=(n, 1), requires_grad=True) * torch.pi / 2
    t = torch.rand(size=(n, 1), requires_grad=True) * torch.pi / 2
    return x.to(device), t.to(device), torch.zeros_like(x).to(device)

# u(x, 0) = cos(x)
def u_x_0(n, device):
    x = torch.rand(size=(n, 1), requires_grad=True) * torch.pi / 2
    t = torch.zeros(size=(n, 1), requires_grad=True)
    return x.to(device), t.to(device), torch.cos(x).to(device)

# ut(x, 0) = -sin(x)
def ut_x_0(n, device):
    x = torch.rand(size=(n, 1), requires_grad=True) * torch.pi / 2
    t = torch.zeros(size=(n, 1), requires_grad=True)
    return x.to(device), t.to(device), -torch.sin(x).to(device)


# auto differential for (partial f) / (partial x)
def differential(f, x, order):
    for _ in range(order):
        f = torch.autograd.grad(outputs=f,
                                inputs=x,
                                grad_outputs=torch.ones_like(f),
                                create_graph=True,
                                retain_graph=True)[0]
    return f


def train(model, epoch, batch_size, device):

    # loss function
    loss_function = torch.nn.MSELoss()

    # L = || utt(x, t) - uxx(x, t) ||^2
    def loss_function_of_control(f, device):
        x, t, constraint = control(batch_size, device)
        u = f(torch.cat([x, t], dim=1))
        return loss_function(differential(u, t, 2) - differential(u, x, 2), constraint)

    # L = || u(x, 0) - cos(x) ||^2
    def loss_function_of_u_x_0(f, device):
        x, t, constraint = u_x_0(batch_size, device)
        u = f(torch.cat([x, t], dim=1))
        return loss_function(u, constraint)

    # L = || ut(x, 0) + sin(x) ||^2
    def loss_function_of_ut_x_0(f, device):
        x, t, constraint = ut_x_0(batch_size, device)
        u = f(torch.cat([x, t], dim=1))
        return loss_function(differential(u, t, 1), constraint)

    # optimization
    optimizer = torch.optim.Adamax(params=model.parameters())
    
    # use amp 
    scaler = GradScaler()

    for i in range(epoch):  

        optimizer.zero_grad()
        
        # use amp 
        with autocast():
          loss = (loss_function_of_control(model, device) + 
                  loss_function_of_u_x_0(model, device) + 
                  loss_function_of_ut_x_0(model, device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        print("epoch = {0}, batch size = {1}, loss rate = {2}\n".format(i, batch_size, loss.item()), end="")


def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=8192, help="训练轮次，和训练步数冲突")
    parser.add_argument("--batch_size", type=int, default=1024, help="每一轮的批次大小")
    parser.add_argument("--device", type=str, default="sdaa", help="指定运行设备")
    parser.add_argument("--show_graph", type=str, default="False", help="查看偏微分方程解的图像")
    args = parser.parse_args()
    
    epoch = args.epoch
    batch_size = args.batch_size
    show_graph = args.show_graph
    
    # use DistributedDataParallel
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"]) # device
    
    torch.sdaa.set_device(local_rank)
    distributed.init_process_group("tccl")

    model = PhysicsInformedNeuralNetwork().to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # training model
    train(model=model, epoch=epoch, batch_size=batch_size, device=local_rank)
    print("training model finished")
    torch.save(model, "wave_equation.pt")

    if show_graph == "True":

        from matplotlib import pyplot
        import numpy

        x = torch.arange(start=0, end=numpy.pi / 2, step=0.01, dtype=torch.float)
        t = torch.arange(start=0, end=numpy.pi / 2, step=0.01, dtype=torch.float)
        X, T = torch.meshgrid(x, t)
        
        # predict
        U = model(torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)).reshape(len(x), len(t)).detach().numpy()
        # real
        V = numpy.cos(X.reshape(-1, 1).numpy() + T.reshape(-1, 1).numpy()).reshape(len(x), len(t))

        print(U)
        print(V)

        fig = pyplot.figure()
        axes3d = pyplot.axes(projection='3d')
        axes3d.plot_surface(X.numpy(), T.numpy(), U, cmap="summer")
        pyplot.show()

        fig = pyplot.figure()
        axes3d = pyplot.axes(projection='3d')
        axes3d.plot_surface(X.numpy(), T.numpy(), V, cmap="autumn")
        pyplot.show()

if __name__ == "__main__":
    main()    