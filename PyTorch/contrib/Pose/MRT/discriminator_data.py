import os
import torch.utils.data as data
import torch
import numpy as np

class D_DATA(data.Dataset):
    def __init__(self, data_path, joints=15):
        ds_path = os.path.join(data_path, 'discriminator_3_120_mocap.npy')
        self.data=np.load(ds_path, allow_pickle=True)
        
        self.len=len(self.data)
        

    def __getitem__(self, index):
        
        input_seq=self.data[index][:,:30,:][:,::2,:]
        output_seq=self.data[index][:,30:,:][:,::2,:]
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq

        

    def __len__(self):
        return self.len

