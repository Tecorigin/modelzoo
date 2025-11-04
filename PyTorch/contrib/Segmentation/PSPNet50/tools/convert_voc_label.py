import os
import numpy as np
from PIL import Image
import argparse

# 定义每个类别的 RGB 颜色和对应的灰度标签值
VOC_COLORMAP = np.array([
    (0, 0, 0),        # 0: background
    (128, 0, 0),      # 1: aeroplane
    (0, 128, 0),      # 2: bicycle
    (128, 128, 0),    # 3: bird
    (0, 0, 128),      # 4: boat
    (128, 0, 128),    # 5: bottle
    (0, 128, 128),    # 6: bus
    (128, 128, 128),  # 7: car
    (64, 0, 0),       # 8: cat
    (192, 0, 0),      # 9: chair
    (64, 128, 0),     # 10: cow
    (192, 128, 0),    # 11: diningtable
    (64, 0, 128),     # 12: dog
    (192, 0, 128),    # 13: horse
    (64, 128, 128),   # 14: motorbike
    (192, 128, 128),  # 15: person
    (0, 64, 0),       # 16: potted plant
    (128, 64, 0),     # 17: sheep
    (0, 192, 0),      # 18: sofa
    (128, 192, 0),    # 19: train
    (0, 64, 128)      # 20: tv/monitor
])

def voc_colormap2label():
    """构建颜色到类别标签值的映射表"""
    colormap2label = np.zeros(256 ** 3)  # 初始化颜色映射表
    for idx, color in enumerate(VOC_COLORMAP):
        colormap2label[(color[0] * 256 + color[1]) * 256 + color[2]] = idx
    return colormap2label

def image2label(image, colormap2label):
    """将 P 模式或 RGB 语义分割图转换为灰度标签图"""
    # 如果图像是 P 模式，将其转换为 RGB
    if image.mode == 'P':
        print(f"图像 {image.filename} 是 P 模式，正在转换为 RGB 模式")
        image = image.convert('RGB')

    # 确保图像是 RGB 模式
    data = np.array(image, dtype='int32')  # 将图片转为 numpy 数组

    # 检查图像的形状，确保它是三通道（RGB）
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"图像 {image.filename} 不是 3 通道 RGB 图像，形状为 {data.shape}")

    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return colormap2label[idx]

def process_segmentation_images(input_dir, output_dir):
    """将指定目录中的 P 模式或 RGB 分割标签图转换为灰度标签图并保存"""
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 构建颜色到标签的映射表
    colormap2label_map = voc_colormap2label()

    # 遍历输入目录中的所有图片
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 打开图片并转换为灰度标签图
            image = Image.open(input_path)
            try:
                label_image = image2label(image, colormap2label_map)

                # 保存灰度标签图
                label_image_pil = Image.fromarray(label_image.astype('uint8'))  # 将 numpy 数组转为 PIL 图像
                label_image_pil.save(output_path)
                print(f"处理完成: {filename}")
            except ValueError as e:
                print(f"错误处理文件 {filename}: {e}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/nvme/common/train_dataset/voc/VOC2007/SegmentationClass', help='input label dir')
    parser.add_argument('--output_dir', type=str, default='/mnt/nvme/common/train_dataset/voc/VOC2007/SegmentationClassGray', help='output gray label dir')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    input_directory = opt.input_dir
    output_directory = opt.output_dir

    # 处理 RGB 语义分割标签图，转换为灰度标签图
    process_segmentation_images(input_directory, output_directory)
