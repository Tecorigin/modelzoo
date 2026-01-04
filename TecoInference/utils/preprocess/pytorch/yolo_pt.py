import cv2
import numpy as np
from PIL import Image

from ...postprocess.pytorch.yolo_pt import letterbox
import torch

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
                                         
def _preprocess_images(img0, input_size, bgr=False):
    # Padded resize
    img = letterbox(img0, input_size, auto=False)[0]

    # Convert
    if bgr:
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    else:
        img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    return img

def preprocess(inputs, batch_size, input_size, batch_padding=True, half=False, **preprocess_parameters):
        # 实现预处理算法
        """
        inputs支持 str, [str], [Image], Image
        """
        image0_shapes = []
        if not isinstance(inputs, (np.ndarray, torch.Tensor)):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            if not batch_padding:
                assert len(inputs) == batch_size, f'images {len(inputs)} != batch_size, you can pass padding=True to resolve it!'

            images = []
            for data in inputs:
                if isinstance(data, str):
                    # 检查是否为图片
                    assert data.split('.')[-1].lower() in IMG_FORMATS, f"{data} is not a image"

                    img0 = cv2.imread(data)  # BGR
                    img = _preprocess_images(img0, input_size, bgr=True)
                elif isinstance(data, Image.Image):
                    img0 = np.asarray(data)
                    img = _preprocess_images(img0, input_size)

                # 检查数据是否为空
                assert img0 is not None, f'Image Not Found {data}'

                images.append(img)
                image0_shapes.append(img0.shape)

            images = np.array(images)
        else:
            images = inputs

        if isinstance(inputs, torch.Tensor):
            images = images.half() if half else images.float()
            images /= 255.

            count = images.shape[0]
            if batch_padding and count < batch_size:
                images = torch.cat([images, torch.zeros([batch_size - count, images.shape[1], images.shape[2], images.shape[3]]).to(images.device, images.dtype)])
            images = images.cpu().numpy()
        else:
            images = images.astype(np.float16) if half else images.astype(np.float32)
            images /= 255

            count = images.shape[0]
            if batch_padding:
                padding = images[None, 0].repeat(batch_size - images.shape[0], 0)
                images = np.concatenate([images, padding])

        return images, count, image0_shapes
