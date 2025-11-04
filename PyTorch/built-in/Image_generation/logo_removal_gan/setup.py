import os
import torch

class Setup(object):
    
    def __init__(self) -> None:

        self.DEVICE = torch.device("cuda:0")
        
        self.clean_dir =  "/data/softws_up/zhaoling/logo-removal-gan/tv-logo/clean"
        self.logo_dir = "/data/softws_up/zhaoling/logo-removal-gan/tv-logo/logo"
        self.patch_size = (256,256)
        self.whole_size = (512,512)
        
        self.BATCH = 2
        self.EPOCHS = 15
        self.GLR = 1e-4
        self.DLR = 4e-4
        self.LAMBDA = 200
        
        self.AUTO = False
        self.BATCH_show = 10


        


        
        