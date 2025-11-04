import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import math
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法

# PSNR 计算函数
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100  # Perfect match
    max_pixel = 1.0  # Assuming normalized images with values in [0, 1]
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# SSIM 计算函数
def calculate_ssim(img1, img2):
    # 如果是批次图像，针对每个图像计算 SSIM
    img1 = img1.cpu().detach().numpy()  # 转换为 NumPy 数组
    img2 = img2.cpu().detach().numpy()  # 转换为 NumPy 数组
    # print("Shape of img1:", img1.shape)
    # print("Shape of img2:", img2.shape)

    ssim_values = []

    # 遍历批次中的每张图像
    for i in range(img1.shape[0]):  # 批次大小
        # 提取单张图片，通道维度应为 [height, width, channels]
        img1_single = img1[i].transpose(1, 2, 0) * 255.0  # 转换为 [height, width, channels]
        img2_single = img2[i].transpose(1, 2, 0) * 255.0  # 转换为 [height, width, channels]
        # print("Shape of img1_single:", img1_single.shape)
        # print("Shape of img2_single:", img2_single.shape)

        # 计算 SSIM
        # 设置 win_size 为 3 或其他适合图像尺寸的值，并设置 data_range
        ssim_value = ssim(img1_single, img2_single, multichannel=True, win_size=3, data_range=255)
        ssim_values.append(ssim_value)

    # 返回批次的平均 SSIM
    return np.mean(ssim_values)

# 网络初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.apply(weights_init)

    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            clean_image = dehaze_net(img_haze)

            loss = criterion(clean_image, img_orig)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

                # Validation Stage
        dehaze_net.eval()  # Set the network to evaluation mode
        psnr_total = 0
        ssim_total = 0
        with torch.no_grad():
            for iter_val, (img_orig, img_haze) in enumerate(val_loader):
                img_orig = img_orig.cuda()
                img_haze = img_haze.cuda()

                clean_image = dehaze_net(img_haze)

                # Calculate PSNR and SSIM for the current batch
                psnr_val = calculate_psnr(clean_image, img_orig)
                ssim_val = calculate_ssim(clean_image, img_orig)

                psnr_total += psnr_val
                ssim_total += ssim_val

                # Save sample output images
                torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0),
                                             config.sample_output_folder + str(iter_val + 1) + ".jpg")

        # Calculate average PSNR and SSIM for the validation set
        avg_psnr = psnr_total / len(val_loader)
        avg_ssim = ssim_total / len(val_loader)

        print(f"Epoch {epoch + 1} - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="original_image/image/")
    parser.add_argument('--hazy_images_path', type=str, default="data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=200)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)
