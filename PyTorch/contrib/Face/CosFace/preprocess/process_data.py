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
import cv2
import argparse
import numpy as np
from mtcnn import MTCNN
from mtcnn.utils.images import load_image


"""
功能：使用 Python + mtcnn 检测人脸并进行五点对齐，得到与 MATLAB 代码等价的结果。

依赖：
  - Python 3.x
  - pip install mtcnn[tensorflow] opencv-python

使用：
  - 通过命令行参数 --data_root 指定数据所在路径。
    其中应包含形如:
       data/CASIA-WebFace/<某个id>/<jpg图像>
       data/lfw/<某个id>/<jpg图像>
  - 通过命令行参数 --result_root 指定输出对齐后图像的保存路径。
  - 运行脚本后，会在 result_root 下生成与原目录结构对应的文件夹，
    如 "CASIA-WebFace-112X96"、"lfw-112X96" 等，每张人脸对齐后的图像保存在对应子目录里。
"""

# --------------------------------------------------------
# 一、参数与参考关键点设置
# --------------------------------------------------------
# 目标人脸对齐后图像大小：高=112, 宽=96
IMG_SIZE = (112, 96)  # (height, width)

# 参考五点坐标（与 MATLAB 脚本保持一致）
coord5point = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)


# --------------------------------------------------------
# 二、辅助函数：收集数据、对齐变换等
# --------------------------------------------------------
def collect_data(folder, dataset_name):
    """
    功能：收集指定目录下的所有 .jpg 图片路径（仅遍历一层子目录）
    参数：
        folder: 如 .../data/CASIA-WebFace
        dataset_name: 数据集名称（如 'CASIA-WebFace'）
    返回：形如 [{'file': xxx, 'dataset': dataset_name}, ...]
    """
    if not os.path.isdir(folder):
        return []

    subfolders = [f for f in os.listdir(folder)
                  if os.path.isdir(os.path.join(folder, f))]
    data_list = []
    for i, sub in enumerate(subfolders):
        sub_path = os.path.join(folder, sub)
        # 只搜集 .jpg 文件，可按需扩展
        jpg_files = [f for f in os.listdir(sub_path) if f.lower().endswith('.jpg')]
        for jf in jpg_files:
            file_path = os.path.join(sub_path, jf)
            data_list.append({'file': file_path, 'dataset': dataset_name})

    return data_list


def align_face(image, src_landmarks, dst_landmarks, output_size):
    """
    功能：对图像进行相似性变换，将 src_landmarks 对齐到 dst_landmarks
         最终输出大小为 output_size=(h, w)。
    参数：
        image: 原始图像, shape=(H, W, C)
        src_landmarks: 检测到的人脸五点, shape=(5, 2), [ [x1,y1], [x2,y2], ... ]
        dst_landmarks: 对齐后的五点坐标参考值, shape=(5, 2)
        output_size: (height, width)
    返回：
        aligned_img: 变换后的人脸图, shape=(height, width, C)
    """
    src_landmarks = src_landmarks.astype(np.float32)
    dst_landmarks = dst_landmarks.astype(np.float32)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(src_landmarks, dst_landmarks, method=cv2.LMEDS)

    # 执行仿射变换
    aligned_img = cv2.warpAffine(
        image,
        M,
        (output_size[1], output_size[0]),  # (width, height)
        borderValue=(0, 0, 0)
    )
    return aligned_img


def get_center_distance(bbox, img_center):
    """
    功能：计算检测到的人脸框中心点 与 图像中心点的距离
    参数：
        bbox: mtcnn返回的框信息, 格式示例: {'x':x, 'y':y, 'width':w, 'height':h}
        img_center: (cx, cy)
    返回：欧式距离的平方
    """
    bx = bbox[0] + bbox[2] / 2.0
    by = bbox[1] + bbox[3] / 2.0
    return (bx - img_center[0]) ** 2 + (by - img_center[1]) ** 2


# --------------------------------------------------------
# 三、主流程
# --------------------------------------------------------
def main(data_root, result_root):
    # 1) 收集 CASIA-WebFace 和 lfw 的数据列表
    casia_folder = os.path.join(data_root, 'CASIA-WebFace')
    lfw_folder = os.path.join(data_root, 'lfw')
    train_list = collect_data(casia_folder, 'CASIA-WebFace')
    test_list = collect_data(lfw_folder, 'lfw')

    data_list = train_list + test_list

    # 2) 初始化 MTCNN 检测器
    detector = MTCNN(device="CPU:0")

    # 3) 遍历所有图像，进行检测与对齐
    for i, item in enumerate(data_list):
        img_path = item['file']
        dataset_name = item['dataset']
        if i % 100 == 0:
            print(f"Processing {i + 1}/{len(data_list)}: {img_path}")

        if not os.path.isfile(img_path):
            continue

        # 读取图像
        img_rgb = load_image(img_path)
        if img_rgb is None or img_rgb.size == 0:
            continue

        # 检测人脸
        detections = detector.detect_faces(img_rgb)

        if len(detections) == 0:
            # 未检测到人脸，跳过
            continue
        elif len(detections) == 1:
            # 只有一张人脸
            face = detections[0]
        else:
            # 多张人脸，选取离图像中心最近的那个
            h, w, _ = img_rgb.shape
            center = (w / 2.0, h / 2.0)
            detections.sort(key=lambda x: get_center_distance(x['box'], center))
            face = detections[0]

        # 提取五点关键点 (顺序与 MATLAB 中一致)
        k = face['keypoints']
        facial5point = np.array([
            [k['left_eye'][0], k['left_eye'][1]],
            [k['right_eye'][0], k['right_eye'][1]],
            [k['nose'][0], k['nose'][1]],
            [k['mouth_left'][0], k['mouth_left'][1]],
            [k['mouth_right'][0], k['mouth_right'][1]]
        ], dtype=np.float32)

        # 将 RGB 转回 BGR，然后再进行对齐和保存。
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.uint8)

        # 4) 进行对齐
        aligned = align_face(img_bgr, facial5point, coord5point, IMG_SIZE)

        # 5) 保存结果：
        #    (a) 将 data_root 替换为 result_root
        #    (b) 把 dataset_name (例如 'CASIA-WebFace') 改成 'CASIA-WebFace-112X96'
        sPathStr, filename = os.path.split(img_path)  # 分离目录与文件名
        name, ext = os.path.splitext(filename)        # 分离文件名和后缀

        # 先把原始的 data_root 路径替换成 result_root，
        tPathStr = sPathStr.replace(data_root, result_root)
        # 进一步把数据集名称替换成 112X96 的后缀
        new_dataset_name = dataset_name + '-112X96'
        tPathStr = tPathStr.replace(dataset_name, new_dataset_name)

        # 如果目标目录不存在，创建之
        if not os.path.exists(tPathStr):
            os.makedirs(tPathStr)

        out_jpg_path = os.path.join(tPathStr, name + '.jpg')
        cv2.imwrite(out_jpg_path, aligned)

    print("全部处理完成。")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 MTCNN 对人脸进行五点对齐的脚本"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/datasets/CosFace/data",
        help="数据所在根目录"
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default="/data/datasets/CosFace/result",
        help="结果输出根目录"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.data_root, args.result_root)
