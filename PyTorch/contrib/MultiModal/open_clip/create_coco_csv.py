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
import json
import os
import csv
import glob

def create_coco_csv(
    annotations_file="/data/teco-data/COCO/annotations/captions_train2017.json",
    images_dir="/data/teco-data/COCO/train2017",
    output_csv="/data/teco-data/COCO/train_data.csv"
):
    # 加载标注文件
    print(f"Loading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # 创建图像ID到文件名的映射
    image_id_to_filename = {}
    for image in annotations['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # 创建图像ID到标题的映射（一个图像可能有多个标题）
    image_id_to_captions = {}
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id not in image_id_to_captions:
            image_id_to_captions[image_id] = []
        image_id_to_captions[image_id].append(caption)
    
    # 验证图像文件是否存在
    print(f"Verifying image files in {images_dir}...")
    available_images = set(os.path.basename(img) for img in glob.glob(os.path.join(images_dir, "*.jpg")))
    
    # 创建CSV文件
    print(f"Creating CSV file at {output_csv}...")
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'title']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 写入数据
        count = 0
        for image_id, filename in image_id_to_filename.items():
            # 检查图像是否存在
            if filename not in available_images:
                continue
                
            # 获取该图像的所有标题
            if image_id not in image_id_to_captions:
                continue
                
            captions = image_id_to_captions[image_id]
            
            # 为每个标题创建一行
            for caption in captions:
                filepath = os.path.join(images_dir, filename)
                writer.writerow({
                    'filepath': filepath,
                    'title': caption
                })
                count += 1
        
        print(f"Successfully wrote {count} entries to {output_csv}")

if __name__ == "__main__":
    create_coco_csv()
    print("CSV file creation completed!")