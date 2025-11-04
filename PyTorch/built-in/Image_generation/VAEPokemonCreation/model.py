import torch
# from matplotlib import pyplot as plt
# from torch.nn import functional as F
from torch import nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            in_channels = out_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_
        x = self.relu(x)
        

class DeResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeResnetBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels = in_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.ConvTranspose2d(
            in_channels = out_channel,
            out_channels = out_channel,
            kernel_size = kernel_size,
            stride = stride,
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x_ = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += x_
        x = self.relu(x)


class VAE(nn.Module):
    def __init__(self, 
                 input_size = 20,
                 hidden_size= 512,
                 lattent_size = 32
                 ):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.lattent_size = lattent_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * input_size * 3, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, lattent_size * 2),
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(lattent_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, input_size * input_size * 3)
        )
        
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
        
    def encode(self, img):
        """
        `return z, mu, log_var`
        """
        self.input_shape = [img.shape[0],  img.shape[1],  img.shape[2],  img.shape[3]]
        lattent_vectors = self.encoder(img)
        mu, log_var = torch.chunk(lattent_vectors, chunks=2, dim=1) # split it
        # log_var  == $\log \sigma^2$
        
        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var
    
    def decode(self, z):
        B, D = z.shape
        self.input_shape[0] = B 
        
        
        decode_img = self.decoder(z)
        decode_img = torch.reshape(decode_img, self.input_shape)
        decode_img = torch.sigmoid(decode_img)
        return decode_img

    def forward(self, img):
        z, mu, log_var = self.encode(img)
        decode_img = self.decode(z)
        return decode_img, mu, log_var
    
   
if __name__ == "__main__":
    img = torch.randn((2, 3, 40, 40))
    print(img.shape)
    model = VAE(40)
    output = model(img)
    print(output) 