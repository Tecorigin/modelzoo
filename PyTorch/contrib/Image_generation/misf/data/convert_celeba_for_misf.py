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
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import os
import zipfile
import shutil
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# ==================== 路径设置 ====================
zip_path = '/data/teco-data/CelebA-MISF/CelebA/Img/img_align_celeba.zip'
mask_root = '/data/teco-data/CelebA-MISF/mask_divide'
output_root = '/data/teco-data/CelebA-MISF'
split_txt = {
    'train': '/data/bigc-data/lcs/misf/dataset/CelebA/Eval/list_eval_partition.txt'
}
valid_ratio = 0.02  # 从 train 中划出 2% 做 valid

==================== 解压 ZIP JPG 图像 ====================
print("🧩 解压 ZIP 图像中...")
unzip_dir = os.path.join(output_root, 'all_jpg')
os.makedirs(unzip_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# ==================== 按 split 划分图像 + 转换为 PNG ====================
print("📁 正在划分图像集并转换为 PNG...")
os.makedirs(output_root, exist_ok=True)
split_file = split_txt['train']
split_map = {'0': 'train', '1': 'valid', '2': 'test'}

split_count = {'train': [], 'valid': [], 'test': []}
valid_counter = 0

with open(split_file, 'r') as f:
    lines = f.readlines()

for line in tqdm(lines):
    img_name, split_id = line.strip().split()
    src_path = os.path.join(unzip_dir,"img_align_celeba/" + img_name)
    target_split = split_map[split_id]

    # 从训练集中划一部分作为 valid
    if target_split == 'train' and valid_counter < valid_ratio * 162770:
        target_split = 'valid'
        valid_counter += 1

    dst_dir = os.path.join(output_root, target_split)
    os.makedirs(dst_dir, exist_ok=True)

    new_name = f"{len(split_count[target_split]) + 1}-1.png"
    dst_path = os.path.join(dst_dir, new_name)

    # 转换为 PNG 并保存
    img = Image.open(src_path).convert("RGB")
    img.save(dst_path, format='PNG')

    split_count[target_split].append(new_name)

# ==================== 处理 Mask 文件夹 ====================
print("🎭 正在复制 mask...")
mask_output = {
    'train': 'mask-train',
    'valid': 'mask-valid',
    'test': 'mask-test'
}

for split in ['train', 'valid']:
    mask_dir = os.path.join(output_root, mask_output[split])
    os.makedirs(mask_dir, exist_ok=True)
    src_mask_dir = os.path.join(mask_root, split)
    for idx, fname in enumerate(sorted(os.listdir(src_mask_dir))):
        new_name = f"{idx + 1}-1.png"
        shutil.copyfile(os.path.join(src_mask_dir, fname),
                        os.path.join(mask_dir, new_name))

# 测试集特殊处理
print("🎭 正在整理测试集 mask...")
mask_test_root = os.path.join(output_root, 'mask-test')
os.makedirs(mask_test_root, exist_ok=True)
subdirs = ['test_20', 'test_40', 'test_60']
range_map = {'test_20': '0%-20%', 'test_40': '20%-40%', 'test_60': '40%-60%'}

for sub in subdirs:
    dst_sub = os.path.join(mask_test_root, range_map[sub])
    os.makedirs(dst_sub, exist_ok=True)
    src_dir = os.path.join(mask_root, sub)
    for idx, fname in enumerate(sorted(os.listdir(src_dir))):
        new_name = f"{idx + 1}-1.png"
        shutil.copyfile(os.path.join(src_dir, fname),
                        os.path.join(dst_sub, new_name))

print("✅ 转换完成！图像已为 PNG，命名符合 MISF 要求。")
