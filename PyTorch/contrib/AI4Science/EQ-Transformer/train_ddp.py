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

import os
import torch, torch_sdaa
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from eq_dataset import Earthquake_data 
from model_transformer import get_model 
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity 

logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True, type=str, default="cuda")
    parser.add_argument("--d_model", type=int, default=64, help="feature dim of transf")
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1500, help="max length of position encoding in transformer")
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument("--epochs", required=True, type=int, default=5, help='train iteration')
    parser.add_argument("--ddp", action='store_true', help="Use DistributedDataParallel")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (GPUs to use)")
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    return parser.parse_args()

def setup(rank, world_size):
    import torch_sdaa.core.sdaa_model as sm
    sm.set_device(rank)
    dist.init_process_group("tccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    model = get_model(args).to(rank)
    train_dataset = Earthquake_data()

    if args.ddp:
        setup(rank, world_size)
        device = torch.device(f'sdaa:{rank}')
        ddp_model = DDP(model)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    else:
        device = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
        ddp_model = model
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_function = nn.MSELoss().to(device)
    loss_seq = []

    for epoch in range(args.epochs):
        global_step = [epoch + 1, 0]

        ddp_model.train()
        total_loss = 0.0

        for batch_idx, (obs_data, eq_data) in enumerate(train_loader):
            global_step[1] = batch_idx + 1
            
            obs_data = obs_data.to(device)
            eq_data = eq_data.to(device)

            out = ddp_model(obs_data)
            loss = loss_function(out, eq_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loss_value = min(loss.item(), 0.1)
            loss_seq.append(loss_value)

            if (batch_idx + 1) % 11 == 0 or (batch_idx + 1) == len(train_loader):
                logger.log(
                    step=global_step,
                    data={"batch": batch_idx + 1, "total_batches": len(train_loader), "loss": loss.item()},
                    verbosity=Verbosity.DEFAULT,
                )

        avg_loss = total_loss / len(train_loader)
        logger.log(
            step=global_step,
            data={"epoch": epoch + 1, "total_epochs": args.epochs, "avg_loss": avg_loss},
            verbosity=Verbosity.DEFAULT,
        )


    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_model_path = os.path.join(script_dir, './model_transf/model.pkl') 
        model_path = os.path.dirname(save_model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if args.ddp:
            torch.save(ddp_model.module.state_dict(), save_model_path)
        else:
            torch.save(model.state_dict(), save_model_path)

        save_loss_path = os.path.join(script_dir, './loss_visible/plt_loss.png')
        loss_path = os.path.dirname(save_loss_path)
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)

        plt.figure()
        plt.plot(loss_seq, color='m')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.show()
        plt.savefig(save_loss_path)
        plt.close()

    if args.ddp:
        cleanup()

if __name__ == "__main__":
    args = parse_arguments()
    if args.ddp:
        args.world_size = args.nproc_per_node * args.nnodes
        mp.spawn(train, args=(args.world_size, args), nprocs=args.nproc_per_node, join=True)
    else:
        train(0, 1, args)
