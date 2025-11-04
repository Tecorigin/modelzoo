import matplotlib.pyplot as plt
import torch
import torch._utils
import torch.nn as nn
from dataset import *
from setup import *


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        
        super(Generator, self).__init__()
        self.down_conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv3 = nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv4 = nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv5 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv6 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.down_conv7 = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.bottleneck = nn.Conv2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv1 = nn.ConvTranspose2d(num_filters*8, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv2 = nn.ConvTranspose2d(num_filters*8*2, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv3 = nn.ConvTranspose2d(num_filters*8*2, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv4 = nn.ConvTranspose2d(num_filters*8*2, num_filters*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv5 = nn.ConvTranspose2d(num_filters*8*2, num_filters*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv6 = nn.ConvTranspose2d(num_filters*4*2, num_filters*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_conv7 = nn.ConvTranspose2d(num_filters*2*2, num_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.out_conv = nn.ConvTranspose2d(num_filters*2, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        
        for param in list(self.parameters()):
            param.requires_grad_(True)

    def forward(self, x):
        
        #print('\nGENERATOR NET:')
        #print('Original: ', x.size())
        # Downsample
        d1 = self.relu(self.down_conv1(x))
        #print('D1: ',d1.size())
        d2 = self.relu(self.down_conv2(d1))
        #print('D2: ',d2.size())
        d3 = self.relu(self.down_conv3(d2))
        #print('D3: ',d3.size())
        d4 = self.relu(self.down_conv4(d3))
        #print('D4: ',d4.size())
        d5 = self.relu(self.down_conv5(d4))
        #print('D5: ',d5.size())
        d6 = self.relu(self.down_conv6(d5))
        #print('D6: ',d6.size())
        d7 = self.relu(self.down_conv7(d6))
        #print('D7: ',d7.size())

        # Bottleneck
        b = self.relu(self.bottleneck(d7))
        #print('B: ',b.size())

        # Upsample
        u1 = self.relu(self.up_conv1(b))
        #print('U1: ',u1.size())
        u2 = self.relu(self.up_conv2(torch.cat([u1, d7], dim=1)))
        #print('U2: ',u2.size())
        u3 = self.relu(self.up_conv3(torch.cat([u2, d6], dim=1)))
        #print('U3: ',u3.size())
        u4 = self.relu(self.up_conv4(torch.cat([u3, d5], dim=1)))
        #print('U4: ',u4.size())
        u5 = self.relu(self.up_conv5(torch.cat([u4, d4], dim=1)))
        #print('U5: ',u5.size())
        u6 = self.relu(self.up_conv6(torch.cat([u5, d3], dim=1)))
        #print('U6: ',u6.size())
        u7 = self.relu(self.up_conv7(torch.cat([u6, d2], dim=1)))
        #print('U7: ',u7.size())

        # Output
        out = self.tanh(self.out_conv(torch.cat([u7, d1], dim=1)))
        #print('Out: ',out.size())

        return out


class Discriminator(nn.Module):
    def __init__(self):
        
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        for param in list(self.parameters()):
            param.requires_grad_(True)
        
    def forward(self, x):
        
        #print('\nDISCRIMINATOR NET:')
        #print('Original: ', x.size())
        x = self.conv1(x)
        #print('Conv1: ', x.size())
        x = self.conv2(x)
        #print('Conv2: ', x.size())
        x = self.conv3(x)
        #print('Conv3: ', x.size())
        x = self.conv4(x)
        #print('Conv5: ', x.size())
        x = self.conv5(x)
        #print('Conv5: ', x.size())
        x = torch.mean(x, dim=[2,3]) 
        #print('Out: ', x.size())
        return x


if __name__ == '__main__':
    
    setup = Setup()
    device = setup.DEVICE

    train_logo_paths, val_logo_paths, train_clean_paths, val_clean_paths = get_paths()
    train_dataset = Dataset(train_logo_paths, train_clean_paths, patches=True)
    train_loader = get_data_loader(train_dataset, batch_size = setup.BATCH)
        
    logos, cleans = next(iter(train_loader))
    if train_dataset.patches_bool:
        num_patches = len(logos)
        logos = torch.cat(logos, dim=0)
        cleans = torch.cat(cleans, dim=0)
    else:
        num_patches = 1
    
    generator = Generator()
    fake_images = generator.forward(logos)

    discriminator = Discriminator()
    output = discriminator.forward(fake_images)
    output2 = discriminator.forward(cleans)

    
    # Print info
    print('Batch size: ', setup.BATCH)
    print('Number of patches for each image: ', num_patches)
    print('Logos size: ',logos.size())                                      # (batch*num_patches, 3, 256, 256)
    print('Fake images size: ',fake_images.size())                          # (batch*num_patches, 3, 256, 256)
    print('Clean image size: ', cleans.size())                              # (batch*num_patches, 3, 256, 256)
    print('Discriminator output of generated size: ', output.size())        # (batch*num_patches, 1)
    print('Discriminator output of cleans size: ', output2.size())          # (batch*num_patches, 1)
    
    # Visualize
    for logo, fake in zip(logos, fake_images):
        logo = denormalize(logo)
        fake = denormalize(fake)
        _, ax = plt.subplots(1,2, figsize=(20,10))
        ax[0].imshow(logo)
        ax[0].title.set_text('Logo')
        ax[1].imshow(fake)
        ax[1].title.set_text('Output')
        plt.show()
        plt.pause(1)
    
    