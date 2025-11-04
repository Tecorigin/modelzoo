import os
import numpy as np

def preprocess(images, half=False):
    images = images.half() if half else images.float()  # uint8 to fp16/32
    images /= 255  # 0 - 255 to 0.0 - 1.0
    images = images.numpy()
    images = images.astype(np.float16) if half else images.astype(np.float32)
    # return np.ascontiguousarray(images)
    return images