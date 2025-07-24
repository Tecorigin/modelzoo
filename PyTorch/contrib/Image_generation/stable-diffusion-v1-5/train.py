import os
import time
from datetime import datetime
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

from torch.sdaa import amp
import torch.sdaa

from run_scripts.argument import parse_args
args = parse_args()

# ==== 设备设置 ====
DEVICE = torch.device(args.device)

# ==== transforms ====
image_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ==== 统一格式logger ====
def write_log(epoch, iteration, loss, ips, total_time, timestamp):
    log_line = (f"TCAPPDLL {timestamp} - Epoch: {epoch} Iteration: {iteration}  rank : 0  "
                f"train.loss : {loss:.6f}  train.ips : {ips:.3f} imgs/s train.total_time : {total_time:.6f}\n")
    with open(args.log_path, "a") as f:
        f.write(log_line)

# ==== 加载模型 ====
def load_model():
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(args.sd_path, "tokenizer"))
    text_encoder = CLIPTextModel.from_pretrained(os.path.join(args.sd_path, "text_encoder"))
    vae = AutoencoderKL.from_pretrained(os.path.join(args.sd_path, "vae"))
    unet = UNet2DConditionModel.from_pretrained(os.path.join(args.sd_path, "unet"))
    scheduler = DDPMScheduler.from_pretrained(os.path.join(args.sd_path, "scheduler"))

    text_encoder.to(DEVICE).eval()
    vae.to(DEVICE).eval()
    unet.to(DEVICE).train()

    for name, param in unet.named_parameters():
        if "mid_block" in name or "out_proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return tokenizer, text_encoder, vae, unet, scheduler

# ==== 数据集加载 ====
def collate_fn(batch):
    images, captions = zip(*batch)
    prompts = [cap[0] for cap in captions]
    return list(images), prompts

def get_loader():
    dataset = CocoCaptions(root=args.coco_img_root, annFile=args.coco_ann_path, transform=image_transforms)
    subset_indices = list(range(args.data_size))
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

# ==== 主训练函数 ====
def train():
    tokenizer, text_encoder, vae, unet, noise_scheduler = load_model()
    dataloader = get_loader()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-4)
    scaler = amp.GradScaler()

    iteration = 0
    if os.path.exists(args.log_path):
        os.remove(args.log_path)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(999):
        for step, (images, prompts) in enumerate(dataloader):
            if iteration >= args.max_iter:
                break

            pixel_values = torch.stack(images).to(DEVICE)

            with torch.no_grad():
                latent_dist = vae.encode(pixel_values).latent_dist
                latents = latent_dist.sample().to(DEVICE) * 0.18215
                text_input = tokenizer(prompts, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEVICE)
                encoder_hidden_states = text_encoder(text_input).last_hidden_state
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=DEVICE).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            start_time = time.time()

            with amp.autocast():
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

            total_time = time.time() - start_time
            ips = args.batch_size / total_time
            timestamp = datetime.now().isoformat(sep=" ", timespec="microseconds")

            print(f"TCAPPDLL {timestamp} - Epoch: {epoch} Iteration: {iteration}  rank : 0  "
                  f"train.loss : {loss.item() * args.accum_steps:.6f}  train.ips : {ips:.3f} imgs/s train.total_time : {total_time:.6f}")
            write_log(epoch, iteration, loss.item() * args.accum_steps, ips, total_time, timestamp)

            iteration += 1

        gc.collect()
        torch.cuda.empty_cache()

        if iteration >= args.max_iter:
            print(f"训练完成，保存模型至 {args.save_path}")
            torch.save(unet.state_dict(), args.save_path)
            break

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.sdaa.set_device(DEVICE)
    train()
