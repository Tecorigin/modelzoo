from torch.utils.data import Dataset
import torch
import random
import numpy as np
import os
import cv2
from PIL import Image
    
class figure_dataset(Dataset):
    def __init__(self, path):
        self.img_name_list = os.listdir(path)
        for i in self.img_name_list:
            assert i.endswith("png"), f"There is a image not ends with png: {i}"
        self.path = path
    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.img_name_list[index])
        image = Image.open(img_name)
        image = image.convert('RGBA') # remove the alpha channel
        image = np.array(image) / 255
        image = torch.Tensor(image)  # H, W, C   C=RGBA
        alpha = image[:, :, -1].unsqueeze(2)  # (H, W, 1)
        image = image[:,:,: 3]
        image = image * alpha +  (1-alpha)
        return image.permute(2, 0, 1) # C ,H, W
    def __len__(self):
        return len(self.img_name_list)
    
if __name__ == "__main__":
    # data = custom_dataset("pixel_color.txt")
    data = figure_dataset("/data/softws_up/zhaoling/modelzoo/PyTorch/built-in/Image_generation/VAEPokemonCreation/figure")
    
    img = data[0]
    print(img.shape)
    
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=np.inf)
    print(img.numpy().shape)
    plt.imshow(img.permute(1, 2, 0).numpy())
    
    plt.savefig("test.jpg")
    # import pdb
    # pdb.set_trace()
    