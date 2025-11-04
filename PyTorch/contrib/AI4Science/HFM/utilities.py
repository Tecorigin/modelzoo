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
import torch
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import warnings
import torch.nn.functional as F
import math
import torch.nn.utils.weight_norm as weight_norm
import  torch.autograd as autograd

loss = nn.MSELoss()
def relative_error(pred, exact, eps=1e-5):
    if type(pred) != type(exact):
        pred = pred.detach().cpu().numpy() if type(pred)==torch.Tensor else pred
        exact = exact.detach().cpu().numpy() if type(exact)==torch.Tensor else exact
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact))/(np.mean(np.square(exact - np.mean(exact)))+eps))
    return torch.sqrt(torch.mean(torch.square(pred - exact))/(torch.mean(torch.square(exact - torch.mean(exact)))+eps))

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return loss(pred, exact)

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, grad_outputs=dummy, create_graph=True)[0]
    return G

def swish(x):
    return x * torch.sigmoid(x)

class neural_net(nn.Module):
    def __init__(self, layers,X=None,**kwargs):
        super().__init__()
        self.num_layers = len(layers)
        self.device = kwargs['device'] if 'device' in kwargs else 0
        temp = []
        self.X = X
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(self.device)
        if self.X is not None:
            self.X_mean = Variable(torch.from_numpy(X.mean(0, keepdims=True)).float(), requires_grad = False).to(self.device)
            self.X_std = Variable(torch.from_numpy(X.std(0, keepdims=True)).float(), requires_grad = False).to(self.device)
        else:
            self.X_mean = Variable(torch.from_numpy(np.array([0.594, 15.04310947,  2.95782569,  2.95816666])).float(), requires_grad = False).to(self.device)#suboff all
            self.X_std = Variable(torch.from_numpy(np.array([1.73181, 8.79611462, 1.76609276, 1.76605707])).float(), requires_grad = False).to(self.device)
            # self.X_mean = Variable(torch.from_numpy(np.array([0.594, 26.96, 2.96, 2.96])).float(), requires_grad = False).to(self.device)#suboff small
            # self.X_std = Variable(torch.from_numpy(np.array([1.73181, 0.3464, 1.7318, 1.7318])).float(), requires_grad = False).to(self.device)
            # self.X_mean = Variable(torch.from_numpy(np.array([8.0, 4.5, 0.0, 5.0])).float(), requires_grad = False).to(self.device) #Cylinder3D
            # self.X_std = Variable(torch.from_numpy(np.array([4.642, 2.0493, 1.4719, 1.4719])).float(), requires_grad = False).to(self.device)
        for l in range(1, self.num_layers):
            temp.append(weight_norm(nn.Linear(layers[l-1], layers[l]), dim = 0))
            nn.init.normal_(temp[l-1].weight)
        self.layers = nn.ModuleList(temp)

        
        print(self.layers)
        #sys.stdout.flush()
        
    def forward(self, x):
        x = ((x - self.X_mean) / self.X_std) # z-score norm
        for i in range(0, self.num_layers-1):
            x = self.layers[i](x)
            if i < self.num_layers-2:
                x = swish(x)
        return x
            
class neural_net2(torch.nn.Module):
    def __init__(self, *, layers, activation='swish',normalization=None, **kwargs):
        super(neural_net2,self).__init__()
        self.layers = layers
        self.num_layers = len(self.layers)
        self.model = nn.ModuleList(
        )
        self.normalization=normalization
        self.Norms = nn.ModuleList() if self.normalization is not None else None
        for l in range(0,self.num_layers-1):
            in_dim = self.layers[l]
            out_dim = self.layers[l+1]
            self.model.append(weight_norm(nn.Linear(in_dim, out_dim,bias=True),dim=0))
            if self.normalization=='BN':
                self.Norms.append(nn.BatchNorm1d(out_dim))
            elif self.normalization=='LN':
                self.Norms.append(nn.LayerNorm(out_dim))
        self.activation=activation
        #初始化模型参数
        for m in self.modules():
             if isinstance(m, nn.Linear):
                 nn.init.normal_(m.weight)  # kaiming高斯初始化
                 nn.init.constant_(m.bias, 0)
    def forward(self, inputs):
        # H = (torch.concat(inputs, 1) - self.X_mean)/self.X_std
        H = inputs
        for l in range(0, len(self.model)):
            H = self.model[l](H)
            if self.normalization is not None:
                H = self.Norms[l](H)
            if l < len(self.model)-2:
                if self.activation=='swish':
                    H =H*torch.sigmoid(H)
                elif self.activation=='relu':
                    H =  F.relu6(H)
                elif self.activation=='tanh':
                    H = F.tanh(H)
                else:
                    raise NotImplementedError
        return H

def Navier_Stokes_2D(c, u, v, p, t, x, y, Pec, Rey):
    
    Y = torch.concat([c, u, v, p], 1)
    # t.requires_grad=True
    # x.requires_grad=True
    # y.requires_grad=True
    Y_t = fwd_gradients(Y, t)
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_xx = fwd_gradients(Y_x, x)
    Y_yy = fwd_gradients(Y_y, y)
    
    c = Y[:,0:1]
    u = Y[:,1:2]
    v = Y[:,2:3]
    p = Y[:,3:4]
    
    c_t = Y_t[:,0:1]
    u_t = Y_t[:,1:2]
    v_t = Y_t[:,2:3]
    
    c_x = Y_x[:,0:1]
    u_x = Y_x[:,1:2]
    v_x = Y_x[:,2:3]
    p_x = Y_x[:,3:4]
    
    c_y = Y_y[:,0:1]
    u_y = Y_y[:,1:2]
    v_y = Y_y[:,2:3]
    p_y = Y_y[:,3:4]
    
    c_xx = Y_xx[:,0:1]
    u_xx = Y_xx[:,1:2]
    v_xx = Y_xx[:,2:3]
    
    c_yy = Y_yy[:,0:1]
    u_yy = Y_yy[:,1:2]
    v_yy = Y_yy[:,2:3]
    
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy) 
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
    
    return e1, e2, e3, e4

def Gradient_Velocity_2D(u, v, x, y):
    
    Y = torch.concat([u, v], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    
    return [u_x, v_x, u_y, v_y]

def Strain_Rate_2D(u, v, x, y):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, x, y)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]


def Navier_Stokes_3D_C(c, u, v, w, p, txyz, Pec, Rey):
    # gradients w.r.t each output and all inputs
    c_txyz = fwd_gradients(c, txyz)
    u_txyz = fwd_gradients(u, txyz)
    v_txyz = fwd_gradients(v, txyz)
    w_txyz = fwd_gradients(w, txyz)
    p_txyz = fwd_gradients(p, txyz)
    
    c_t = c_txyz[:,0:1]
    c_x = c_txyz[:,1:2]
    c_y = c_txyz[:,2:3]
    c_z = c_txyz[:,3:4]
    
    u_t = u_txyz[:,0:1]
    u_x = u_txyz[:,1:2]
    u_y = u_txyz[:,2:3]
    u_z = u_txyz[:,3:4]
                           
    v_t = v_txyz[:,0:1]
    v_x = v_txyz[:,1:2]
    v_y = v_txyz[:,2:3]
    v_z = v_txyz[:,3:4]
    
    w_t = w_txyz[:,0:1]
    w_x = w_txyz[:,1:2]
    w_y = w_txyz[:,2:3]
    w_z = w_txyz[:,3:4]
                           
    p_x = p_txyz[:,1:2]
    p_y = p_txyz[:,2:3]
    p_z = p_txyz[:,3:4]

    # second gradient
    
    c_x_txyz = fwd_gradients(c_x, txyz)
    c_y_txyz = fwd_gradients(c_y, txyz)
    c_z_txyz = fwd_gradients(c_z, txyz)
    c_xx = c_x_txyz[:,1:2] #wanted
    c_yy = c_y_txyz[:,2:3] #wanted
    c_zz = c_z_txyz[:,3:4] #wanted
                           
                
    u_x_txyz = fwd_gradients(u_x, txyz)
    u_y_txyz = fwd_gradients(u_y, txyz)
    u_z_txyz = fwd_gradients(u_z, txyz)
    u_xx = u_x_txyz[:,1:2] #wanted
    u_yy = u_y_txyz[:,2:3] #wanted
    u_zz = u_z_txyz[:,3:4] #wanted
    
    v_x_txyz = fwd_gradients(v_x, txyz)
    v_y_txyz = fwd_gradients(v_y, txyz)
    v_z_txyz = fwd_gradients(v_z, txyz)
    v_xx = v_x_txyz[:,1:2] #wanted
    v_yy = v_y_txyz[:,2:3] #wanted
    v_zz = v_z_txyz[:,3:4] #wanted
                           
    w_x_txyz = fwd_gradients(w_x, txyz)
    w_y_txyz = fwd_gradients(w_y, txyz)
    w_z_txyz = fwd_gradients(w_z, txyz)
    w_xx = w_x_txyz[:,1:2] #wanted
    w_yy = w_y_txyz[:,2:3] #wanted
    w_zz = w_z_txyz[:,3:4] #wanted
    
    e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e1, e2, e3, e4, e5

def Navier_Stokes_3D(u, v, w, p, txyz, Pec, Rey):
    # gradients w.r.t each output and all inputs
   
    u_txyz = fwd_gradients(u, txyz)
    v_txyz = fwd_gradients(v, txyz)
    w_txyz = fwd_gradients(w, txyz)
    p_txyz = fwd_gradients(p, txyz)
    
    
    u_t = u_txyz[:,0:1]
    u_x = u_txyz[:,1:2]
    u_y = u_txyz[:,2:3]
    u_z = u_txyz[:,3:4]
                           
    v_t = v_txyz[:,0:1]
    v_x = v_txyz[:,1:2]
    v_y = v_txyz[:,2:3]
    v_z = v_txyz[:,3:4]
    
    w_t = w_txyz[:,0:1]
    w_x = w_txyz[:,1:2]
    w_y = w_txyz[:,2:3]
    w_z = w_txyz[:,3:4]
                           
    p_x = p_txyz[:,1:2]
    p_y = p_txyz[:,2:3]
    p_z = p_txyz[:,3:4]

    # second gradient
                           
    u_x_txyz = fwd_gradients(u_x, txyz)
    u_y_txyz = fwd_gradients(u_y, txyz)
    u_z_txyz = fwd_gradients(u_z, txyz)
    u_xx = u_x_txyz[:,1:2] #wanted
    u_yy = u_y_txyz[:,2:3] #wanted
    u_zz = u_z_txyz[:,3:4] #wanted
    
    v_x_txyz = fwd_gradients(v_x, txyz)
    v_y_txyz = fwd_gradients(v_y, txyz)
    v_z_txyz = fwd_gradients(v_z, txyz)
    v_xx = v_x_txyz[:,1:2] #wanted
    v_yy = v_y_txyz[:,2:3] #wanted
    v_zz = v_z_txyz[:,3:4] #wanted
                           
    w_x_txyz = fwd_gradients(w_x, txyz)
    w_y_txyz = fwd_gradients(w_y, txyz)
    w_z_txyz = fwd_gradients(w_z, txyz)
    w_xx = w_x_txyz[:,1:2] #wanted
    w_yy = w_y_txyz[:,2:3] #wanted
    w_zz = w_z_txyz[:,3:4] #wanted
    
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e2, e3, e4, e5

def Gradient_Velocity_3D(u, v, w, x, y, z):
    
    Y = torch.concat([u, v, w], 1)
    
    Y_x = fwd_gradients(Y, x)
    Y_y = fwd_gradients(Y, y)
    Y_z = fwd_gradients(Y, z)
    
    u_x = Y_x[:,0:1]
    v_x = Y_x[:,1:2]
    w_x = Y_x[:,2:3]
    
    u_y = Y_y[:,0:1]
    v_y = Y_y[:,1:2]
    w_y = Y_y[:,2:3]
    
    u_z = Y_z[:,0:1]
    v_z = Y_z[:,1:2]
    w_z = Y_z[:,2:3]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, x, y, z, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, x, y, z)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz


