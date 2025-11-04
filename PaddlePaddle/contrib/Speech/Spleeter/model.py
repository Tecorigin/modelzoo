import numpy as np 
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def encoder_block(out_channel):
    return nn.Sequential(
        nn.BatchNorm2D(out_channel, epsilon=1e-3, momentum=0.01),
        nn.LeakyReLU(0.2)
    )

def decoder_block(in_channel, out_channel, dropout=False):
    layers = nn.LayerList([
        nn.Conv2DTranspose(in_channel, out_channel, kernel_size=5, stride=2),
        nn.ReLU(),
        nn.BatchNorm2D(out_channel, epsilon=1e-3, momentum=0.01)
    ])
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNet(nn.Layer):
    def __init__(self, in_channel=2):
        super(UNet, self).__init__()

        self.pad  = nn.ZeroPad2D(padding=(1, 2, 1, 2))

        ### Encoder ###
        # First Layer
        self.conv1     =  nn.Conv2D(in_channel, 16, kernel_size=5, stride=2)   ## padding 
        self.encoder1  =  encoder_block(16)
        # Second Layer
        self.conv2     =  nn.Conv2D(16, 32, kernel_size=5, stride=2) 
        self.encoder2  =  encoder_block(32)
        # Third Layer
        self.conv3     =  nn.Conv2D(32, 64, kernel_size=5, stride=2)  
        self.encoder3  =  encoder_block(64)
        # Fourth Layer
        self.conv4     =  nn.Conv2D(64, 128, kernel_size=5, stride=2) 
        self.encoder4  =  encoder_block(128)
        # Fifth Layer
        self.conv5     =  nn.Conv2D(128, 256, kernel_size=5, stride=2) 
        self.encoder5  =  encoder_block(256)
        # Sixth Layer
        self.conv6     =  nn.Conv2D(256, 512, kernel_size=5, stride=2) 
        self.encoder6  =  encoder_block(512)

        ### Decoder ###
        # First Layer
        self.decoder1  =  decoder_block(512, 256, dropout=True)    
        # Second Layer
        self.decoder2  =  decoder_block(512, 128, dropout=True)  # Skip Connection
        # Third Layer
        self.decoder3  =  decoder_block(256, 64, dropout=True)
        # Fourth Layer
        self.decoder4  =  decoder_block(128, 32)
        # Fifth Layer
        self.decoder5  =  decoder_block(64, 16)
        # Sixth Layer
        self.decoder6  =  decoder_block(32, 1)

        # Last Layer
        self.mask      =  nn.Conv2D(1, 2, kernel_size=4, dilation=2, padding=3) 
        self.sig       =  nn.Sigmoid()
        
        

    def forward(self, x, output_mask_logit=False):
        ### Encoder ###
        skip1   =  self.pad(x)
        skip1   =  self.conv1(skip1)
        down1   =  self.encoder1(skip1)
        
        skip2   =  self.pad(down1)
        skip2   =  self.conv2(skip2)
        down2   =  self.encoder2(skip2)

        skip3   =  self.pad(down2)
        skip3   =  self.conv3(skip3)
        down3   =  self.encoder3(skip3)
        
        skip4   =  self.pad(down3)
        skip4   =  self.conv4(skip4)
        down4   =  self.encoder4(skip4)

        skip5   =  self.pad(down4)
        skip5   =  self.conv5(skip5)
        down5   =  self.encoder5(skip5)

        skip6   =  self.pad(down5)
        skip6   =  self.conv6(skip6)
        down6   =  self.encoder6(skip6)

        ### Decoder ###
        up1     =  self.decoder1(skip6)
        up1     =  up1[:, :, 1: -2, 1: -2]   
        merge1  =  paddle.concat((skip5, up1), 1)            # TensorFlow [B * H * W * C] PyTorch [B * C * H * W]
        
        up2     =  self.decoder2(merge1)
        up2     =  up2[:, :, 1: -2, 1: -2] 
        merge2  =  paddle.concat((skip4, up2), 1)

        up3     =  self.decoder3(merge2)
        up3     =  up3[:, :, 1: -2, 1: -2] 
        merge3  =  paddle.concat((skip3, up3), 1)

        up4     =  self.decoder4(merge3)
        up4     =  up4[:, :, 1: -2, 1: -2] 
        merge4  =  paddle.concat((skip2, up4), 1)

        up5     =  self.decoder5(merge4)
        up5     =  up5[:, :, 1: -2, 1: -2] 
        merge5  =  paddle.concat((skip1, up5), 1)

        up6     =  self.decoder6(merge5)
        up6     =  up6[:, :, 1: -2, 1: -2]
        
        m       =  self.mask(up6)
        
        ### Mask Output ###
        if not output_mask_logit:
            m   =  self.sig(m)
            return x * m    # element-wise
        m = self.sig(m)

        return m


class MultiLoss(nn.Layer):
    def __init__(self, model_list, criterion, params):
        super(MultiLoss, self).__init__()
        self.model_list = model_list
        self.num_instrument = params['num_instruments']
        self.criterion = criterion
        #self.weights = [1] * len(self.num_instrument)
        self.sum = len(self.num_instrument)
        self.softmax = nn.Softmax(axis=0) 

    def forward(self, mix_stft_mag, separate_stft_mag):
        loss = 0
        
        pred_stft_mag = []
    
        for i in range(self.sum):
            model = self.model_list[i]
            pred = model(mix_stft_mag)
            pred_stft_mag.append(pred)
        
        # pred_stft_mag = self.softmax(torch.stack(pred_stft_mag, dim=0))

        for i in range(self.sum):
            loss += self.criterion(pred_stft_mag[i] * mix_stft_mag, separate_stft_mag[i])
        loss = loss / self.sum

        return loss



class Softmax_UNet(nn.Layer):
    def __init__(self):
        super(Softmax_UNet, self).__init__()
        self.softmax = nn.Softmax(axis=0)

    def forward(self, masks):
        masks = paddle.stack(masks, axis=0)
        return self.softmax(masks)

