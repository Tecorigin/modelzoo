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
import string
from torch.utils.data import Dataset
import numpy as np
import os
import warnings
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

class Dataset3D(Dataset):

    def __init__(self,  path:string, T_data=None, N_data=None,cols=9,snap=100,eqns=False, 
                other_loss=False, multiple_load=False, save_counts=None):
        super(Dataset3D, self).__init__()
        if snap>=N_data: raise RuntimeError("snap>=N_data")
        logger.info(f"{self} Loading dataset...")
        logger.info(f'args:path:{path},T_data:{T_data},N_data:{N_data},cols:{cols},eqns:{eqns},other_loss:{other_loss}')
        self.head_len = 128
        self.path = path
        self.f = os.open(self.path,os.O_RDONLY)
        logger.info(f'open file: {self.f}')
        self.cols = cols
        self.T_data= T_data #时间维度
        self.N_data = N_data #行数
        self.row_bytes = self.cols*8 #float64占8个字节
        self.snap = snap
        self.eqns=eqns #是否使用生成数据
        self.data_eqns=None
        # self.getdata_eqns() if self.eqns else None
        self.other_loss = other_loss #使用u,v,w,p参与loss计算
        self.multiple_load=multiple_load #如果npy采用多次save保存则必须设为True
        self.save_counts=save_counts #npy一次save的总行数
        assert not multiple_load or save_counts is not None #如果multiple_load为True则save_counts必须准确指定
    def __len__(self):
        return self.N_data-1

    def __getitem__(self, index):
        if self.multiple_load:
            offset = self.head_len*(index//self.save_counts+1) + index*self.cols*8
        else:
            offset = self.head_len + index*self.cols*8
        try:
            buffer = os.pread(self.f, self.row_bytes, offset)
            if buffer is None or len(buffer)==0:
                warnings.warn("empty data! Please check datasets!")
                raise RuntimeError("empty data")
            data = np.frombuffer(buffer, dtype=np.float64)
            data = data.astype(np.float32)
            t, x, y, z, u, v, w, p, c = data
            if self.other_loss:
                data = [t, x, y, z, u, v, w, p, c]
            else:
                data = [t, x, y, z, c]
            if self.eqns:
                data_eqns = self.get_random_data()
                t_eqns, e_qns, y_eqns, z_eqns,*_ = data_eqns
                return data+[t_eqns, e_qns, y_eqns, z_eqns]
        except OSError as e:
            warnings.warn(f'file describor {str(self.f)} is closed!')
            self.f = os.open(self.path,os.O_RDONLY)
            return self.__getitem__(index)
    
    def get_random_data(self):
        index = np.random.randint(self.N_data)
        data = self.get_data_by_index(index)
        return data[:4]
    def get_data_by_index(self, index):
        if self.multiple_load:
            offset = self.head_len*(index//self.save_counts+1) + index*self.cols*8
        else:
            offset = self.head_len + index*self.cols*8
        buffer = os.pread(self.f, self.row_bytes, offset)
        if buffer is None or len(buffer)==0:
            warnings.warn("empty data! Please check datasets!")
            raise RuntimeError("empty data")
        data = np.frombuffer(buffer, dtype=np.float64)
        data = data.astype(np.float32)
        return data

    def close_f(self):
        os.close(self.f)
    
    def getTestData(self, snap, index,end=None):
        #获取某个时间点的全部空间数据
        end=self.N_data if end is None else end
        data=[]
        for idx in range(index, end, snap):
            offset = self.head_len + idx*self.cols*8
            buffer = os.pread(self.f, self.row_bytes, offset)
            row = np.frombuffer(buffer,np.float64)
            data.append(row)
        # if self.toTensor:
        #     data = list(map(lambda x : torch.tensor(x, requires_grad=False, dtype=torch.float32, device=self.device),data))
        data = np.array(data)
        x = data[:,1].reshape(-1,1)
        y = data[:,2].reshape(-1,1)
        z = data[:,3].reshape(-1,1)
        t = data[:,0].reshape(-1,1)
        c = data[:,8].reshape(-1,1)
        u = data[:,4].reshape(-1,1)
        v = data[:,5].reshape(-1,1)
        w = data[:,6].reshape(-1,1)
        p = data[:,7].reshape(-1,1)
        return [x, y, z, t, c, u, v, w, p]
    
    # @staticmethod
    def getdata_eqns(self, t=[1,5,0.08], x=[35, 40, 0.1], y=[5, 15, 0.1], z=[5, 10, 0.1]):
        if self.data_eqns is None:
            T_eqns = np.arange(t[0], t[1], t[2]).reshape(-1, 1)
            X_eqns = np.arange(x[0], x[1], x[2]).reshape(-1, 1)
            Y_eqns = np.arange(y[0], y[1], y[2]).reshape(-1, 1)
            Z_eqns = np.arange(z[0], z[1], z[2]).reshape(-1, 1)
            data = cal_cartesian_coord([T_eqns, X_eqns, Y_eqns, Z_eqns])
            self.data_eqns=data
        return self.data_eqns

def cal_cartesian_coord(arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points

def cal_mean_std(path,T_data,N_data):
    f = os.open(path,os.O_RDONLY)
    std = 0
    mean = 0
    T = []
    X = []
    Y = []
    Z = []
    for ind in range(0, N_data, T_data):
        offset = 128 + ind*9*8
        buffer = os.pread(f, 72, offset)
        if buffer is None or len(buffer)==0:
            raise RuntimeError("empty data")
        data = np.frombuffer(buffer, dtype=np.float64)
        t, x, y, z, u, v, w, p, c  = data
        T.append(t)
        X.append(x)
        Y.append(y)
        Z.append(z)

    inputs = np.concatenate([T,X,Y,Z],1)
    mean = np.mean(inputs,0)
    std = np.std(inputs,0)
    return mean, std

if __name__=='__main__':
    path = '/home/hpc/cfd/Datasets_HFM/Cylinder3D_t201_n184671.npy'
    T_data = 201
    N_data = 184671#63916*T_data
    dataset = Dataset3D(path,T_data,N_data,)
    import pdb;pdb.set_trace()
