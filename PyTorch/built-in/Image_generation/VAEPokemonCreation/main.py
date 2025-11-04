# from datasets.img_reader import colormap, pixel_color
import os
import argparse
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch import nn as nn
from torch import optim as optim 
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from utils.data_utils import figure_dataset
from utils.lr_sched  import adjust_learning_rate
import numpy as np


from model import VAE

def get_args_parser():
    parser = argparse.ArgumentParser('Pokemon Generator by LSTM Training', add_help=True)
    # VAE  (model) settings
    parser.add_argument("--hidden_size", default=1024, type=int,
                        help = 'VAE settings, size of hidden layer for encoder and decoder')
    parser.add_argument("--latent_size", default=16, type=int,
                        help = 'VAE settings, size of the latent vector.')
    
    # training settings
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Batchsize per GPU")
    parser.add_argument("--output_dir", default="output_dir", type= str,
                        help = 'output dir for ckpt and logs')
    parser.add_argument("--epoch", default=10000, type=int,
                        help = 'Number of epochs')
    parser.add_argument("--lr", default="5e-4", type=float,
                        help = 'Learning rate')
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device: cuda or GPU")
    parser.add_argument("--save_period", default=200, type=int,
                        help = 'masked rate of the input images')
    parser.add_argument("--warmup_epochs", default=20, type=int,
                        help = 'warmup epochs')
    parser.add_argument("--min_lr", default=1e-7, type=float,
                        help = 'min lr for lr schedule')
    parser.add_argument("--seed", default=41, type=int,
                        help = 'random seed init')

    return parser

# 计算VAE的损失函数
def vae_loss(reconstructed_x, x, mu, log_var):
    # print(torch.max(x), torch.min(x))
    # print(torch.max(reconstructed_x), torch.min(reconstructed_x))/
    # reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction="mean")
    # reconstruction_loss = F.binary_cross_entropy(reconstructed_x, x)
    
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction="sum")  # The Binary Cross-Entropy (BCE) loss lacks theoretical support.
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / 16
    return reconstruction_loss * 10 + kl_divergence, kl_divergence # 10 is magic number for balance

def main(args):
    # torch.manual_seed(114514)
    output_path = f"{args.output_dir}_lr{args.lr}_epoch{args.epoch}_latent{args.latent_size}_hidden{args.hidden_size}"
    args.output_dir = output_path
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    my_dataset = figure_dataset("/data/softws_up/zhaoling/modelzoo/PyTorch/built-in/Image_generation/VAEPokemonCreation/figure")

    train_ratio = 0.99
    test_ratio = 1 - train_ratio

    train_size = int(train_ratio * len(my_dataset))
    test_size = len(my_dataset) - train_size

    train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])

    print(len(train_dataset))
    print(len(test_dataset))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size = args.batch_size, 
        shuffle=True,
        drop_last=True
    )


    device = args.device
    model = VAE(
        input_size=40,
        hidden_size=args.hidden_size,
        lattent_size=args.latent_size
        )
    
    model = model.to(device)
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    scaler = torch.sdaa.amp.GradScaler()
    #  -----------------------------------
    for epoch in range(args.epoch):
        running_loss = 0.0
        running_kl = 0.0
        
        # train for one epoch
        model.train()
        lr = adjust_learning_rate(optimizer=optimizer, epoch=epoch, args=args)
        log_writer.add_scalar("train/lr", lr, epoch)
        for data in train_loader:
            data = data.to(device)
            # print(data.shape)
            optimizer.zero_grad()
            predict_img, mu, log_var = model(data)
            # print(predict_img.shape)
            loss, kl_divergence = vae_loss(
                    reconstructed_x=predict_img,
                    x = data,
                    mu=mu,
                    log_var=log_var
                )
            # print(loss)
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss
            running_kl += kl_divergence
        log_writer.add_scalar("train/epoch_loss", running_loss / len(train_loader), epoch)
        log_writer.add_scalar("train/epoch_kl_divergence", running_kl / len(train_loader), epoch)
        log_writer.add_scalar("train/epoch_reconstruct_loss", (running_loss - running_kl) / len(train_loader), epoch)
        if epoch % 50 == 0 or epoch == args.epoch:
            log_writer.add_images("train/x", data[:32], epoch)
            log_writer.add_images("train/reconstruct_img", predict_img[:32], epoch)
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        log_writer.flush()
        
        if epoch % args.save_period == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_{epoch}.pt"))
        
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    main(args)
