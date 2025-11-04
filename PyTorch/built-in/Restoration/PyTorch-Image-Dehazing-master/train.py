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
import net_selfattention
import net
import numpy as np
from torchvision import transforms
from torch_sdaa.utils import cuda_migrate  # 使用torch_sdaa自动迁移方法

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度训练所需的模块

def train(config):
    # 初始化模型
    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.apply(weights_init)

    # 加载数据集
    train_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # 定义损失函数和优化器
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 混合精度训练：初始化 GradScaler
    scaler = GradScaler()

    # 训练模式
    dehaze_net.train()

    for epoch in range(config.num_epochs):
        for iteration, (img_orig, img_haze) in enumerate(train_loader):
            img_orig = img_orig.cuda()
            img_haze = img_haze.cuda()

            # 混合精度训练：前向传播
            with autocast():
                clean_image = dehaze_net(img_haze)
                loss = criterion(clean_image, img_orig)

            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 混合精度训练：反向传播
            scaler.unscale_(optimizer)  # 混合精度训练：梯度裁剪前取消缩放
            torch.nn.utils.clip_grad_norm_(dehaze_net.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)  # 混合精度训练：更新参数
            scaler.update()  # 混合精度训练：更新缩放器

            # 每隔 1 个 iteration 打印一次 loss
            print(f"Epoch [{epoch + 1}/{config.num_epochs}], Iteration [{iteration + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 每隔 config.snapshot_iter 个 iteration 保存一次模型
            if (iteration + 1) % config.snapshot_iter == 0:
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + f"Epoch{epoch}_Iter{iteration + 1}.pth")

        # 验证阶段
        dehaze_net.eval()
        with torch.no_grad():
            for iter_val, (img_orig, img_haze) in enumerate(val_loader):
                img_orig = img_orig.cuda()
                img_haze = img_haze.cuda()

                # 混合精度训练：前向传播
                with autocast():
                    clean_image = dehaze_net(img_haze)

                # 保存验证结果
                torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig), 0), config.sample_output_folder + f"val_{iter_val + 1}.jpg")

        # 保存最终模型
        torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth")
        dehaze_net.train()  # 恢复训练模式
# def train(config):

# 	dehaze_net = net.dehaze_net().cuda()
# 	dehaze_net.apply(weights_init)

# 	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
# 											 config.hazy_images_path)		
# 	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
# 											 config.hazy_images_path, mode="val")		
# 	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
# 	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

# 	criterion = nn.MSELoss().cuda()
# 	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
# 	dehaze_net.train()

# 	for epoch in range(config.num_epochs):
# 		for iteration, (img_orig, img_haze) in enumerate(train_loader):

# 			img_orig = img_orig.cuda()
# 			img_haze = img_haze.cuda()

# 			clean_image = dehaze_net(img_haze)

# 			loss = criterion(clean_image, img_orig)

# 			optimizer.zero_grad()
# 			loss.backward()
# 			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
# 			optimizer.step()

# 			if ((iteration+1) % config.display_iter) == 0:
# 				print("Loss at iteration", iteration+1, ":", loss.item())
# 			if ((iteration+1) % config.snapshot_iter) == 0:
				
# 				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		

# 		# Validation Stage
# 		for iter_val, (img_orig, img_haze) in enumerate(val_loader):

# 			img_orig = img_orig.cuda()
# 			img_haze = img_haze.cuda()

# 			clean_image = dehaze_net(img_haze)

# 			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

# 		torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth") 

			








if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="original_image/image/")
	parser.add_argument('--hazy_images_path', type=str, default="data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--train_batch_size', type=int, default=8)
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








	
