import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
from setup import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def get_paths():
    setup = Setup()
    clean_paths = []
    logo_paths = []
    
    for logo_img_name in os.listdir(setup.logo_dir):
            
        # Extract the i-j part of the logo image filename
        clean_filename = logo_img_name.split("-")[0] + "-" + logo_img_name.split("-")[1] + ".jpg"
            
        # Find the path to the corresponding clean image
        clean_path = os.path.join(setup.clean_dir, clean_filename)
            
        # Find the path to the corresponding logo image
        logo_path = os.path.join(setup.logo_dir, logo_img_name)
            
        # Add the mapping to the dictionary
        clean_paths.append(clean_path)
        logo_paths.append(logo_path)

    # Divide into train/val set
    split_point = int(len(logo_paths)*0.7)
    train_clean_paths, val_clean_paths = clean_paths[:split_point], clean_paths[split_point:]
    train_logo_paths, val_logo_paths = logo_paths[:split_point], logo_paths[split_point:]
    
    return train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths


class Dataset(data.Dataset):
    
    def __init__(self, logo_paths:list[str], clean_paths:list[str], patches:bool):
        
        setup = Setup()
        self.patches_bool = patches
        
        if self.patches_bool:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(setup.whole_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]) 

        self.clean_paths = clean_paths
        self.logo_paths = logo_paths


    def __len__(self):
        return len(self.logo_paths)
    
    
    def __getitem__(self, idx):
        
        logo_path = self.logo_paths[idx]
        clean_path = self.clean_paths[idx]
        
        logo_image = Image.open(logo_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        if self.patches_bool:
            logo_patches = self._get_patches(logo_image)
            clean_patches = self._get_patches(clean_image)

            if self.transform:
                        logo_patches = [self.transform(patch) for patch in logo_patches]
                        clean_patches = [self.transform(patch) for patch in clean_patches]
        
            return logo_patches, clean_patches
        
        else:
            logo_tensor = self.transform(logo_image)
            clean_tensor = self.transform(clean_image)
            
            return logo_tensor, clean_tensor           
            

        
    def _get_patches(self, image):
        
        setup = Setup()
        
        # Resize image to multiples of patch_size
        w, h = image.size
        w_multiple = w // setup.patch_size[0]
        h_multiple = h // setup.patch_size[1]
        resize_size = (w_multiple * setup.patch_size[0], h_multiple * setup.patch_size[1])
        image = image.resize(resize_size)
        
        # Extract patches from image
        patches = []
        for i in range(0, w_multiple):
            for j in range(0, h_multiple):
                x = i * setup.patch_size[0]
                y = j * setup.patch_size[1]
                patch = image.crop((x, y, x + setup.patch_size[0], y + setup.patch_size[1]))
                patches.append(patch)
                
        return patches


def denormalize(image: torch.Tensor):
    inv_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()

    
def get_data_loader(dataset:Dataset, batch_size:int):
    return DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = 3,
        pin_memory = True,
        shuffle = True
    )
    
if __name__ == '__main__':
    
        setup = Setup()
        
        train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
        
        print('Size of training set: ', len(train_logo_paths))
        print('Size of validation set: ', len(val_logo_paths))
        
        train_dataset = Dataset(train_logo_paths, train_clean_paths, patches = True)
        train_loader = get_data_loader(train_dataset, batch_size= setup.BATCH_show)
        
        logos, cleans = next(iter(train_loader))
        
        if train_dataset.patches_bool:
            num_patches = len(logos)
            logos = torch.cat(logos, dim=0)
            cleans = torch.cat(cleans, dim=0)
        else:
            num_patches = 1
            
        print('\nSize of logos after concatenation: {}      \
               \nNumber of patches for each image: {}        \
               \nBatch size: {}'.format(logos.size(), num_patches, setup.BATCH_show))

        # logos_concatenated : [batch*patches, 3, 256, 256]
        # cleans_concatenated: [batch*patches, 3, 256, 256]
        # for patch_size = (256,256) --> 10 patches
        
        for logo, clean in zip(logos, cleans):
            logo = denormalize(logo)
            clean = denormalize(clean)

            _, ax = plt.subplots(1,2, figsize=(20,10))
            ax[0].imshow(logo)
            ax[0].title.set_text('Logo')
            ax[1].imshow(clean)
            ax[1].title.set_text('Clean')
            plt.show()
            plt.pause(1)