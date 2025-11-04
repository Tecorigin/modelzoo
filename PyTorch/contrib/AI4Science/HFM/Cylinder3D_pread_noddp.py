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
from pathlib import Path
import torch
import torch_sdaa
import numpy as np
import scipy.io
import time
import sys
from torch.utils.data import DataLoader
import torch.optim as optim
from utilities import neural_net, Navier_Stokes_3D, \
                       mean_squared_error, relative_error, neural_net2
from LoadDataset_pread_ddp import Dataset3D
from memory_profiler import profile 
import psutil
import os
from tqdm import tqdm
from ddp_utils import *
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from model_predict import end_train_wrapper
import argparse
import torch.autograd as autograd
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

class HFM(object):
    
    def __init__(self, layers, batch_size,Pec, Rey,save_path="weights/Cylinder3D.pth", lr=8e-5, other_loss=False, eqns=False, alpha=0.7):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        self.save_path = save_path
        self.lr = lr
        self.other_loss=other_loss
        self.eqns = eqns
        self.alpha = alpha
    # @profile
    def train(self,rank, device, world_size, total_epoch, path,T_data, N_data, ckpt_path=None, activation='swish', normalization=None, args=None):
        dataset = Dataset3D(path=path,snap=100,T_data=T_data,N_data=N_data, eqns=self.eqns, other_loss=self.other_loss)
        self.dataloader_data = DataLoader(dataset, batch_size=self.batch_size//world_size,shuffle=False, \
                                         num_workers=4,multiprocessing_context='fork')
        self.net_cuvwp = neural_net2(layers = self.layers,device=device,activation=activation, normalization=normalization)
        self.net_cuvwp.to(device)
        if rank==0 and ckpt_path is not None and os.path.isfile(ckpt_path):
            self.net_cuvwp.load_state_dict(torch.load(ckpt_path))
        self._params = filter(lambda p: p.requires_grad, self.net_cuvwp.parameters())
        self.optimizer = optim.Adam(self._params,lr=self.lr,betas=(0.9,0.999),eps=1e-6, weight_decay=1e-5,amsgrad=False)
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device)
        start_time = time.time()
        running_time = 0.0
        epoch = 0
        self.net_cuvwp.train()
        while epoch < total_epoch:
            for i, data in enumerate(self.dataloader_data):
                if self.other_loss is False and len(data)<6:
                    data=data + data[:-1]
                if self.other_loss and len(data)<10:
                    data=data+data[0:4]
                data = list(map(lambda x: torch.tensor(x, dtype=torch.float32), data))
                if self.other_loss is False:
                    [t_data_batch, x_data_batch, y_data_batch, z_data_batch, c_data_batch, t_eqns_batch, 
                        x_eqns_batch, y_eqns_batch, z_eqns_batch] = map(lambda x:x.unsqueeze(-1),data)
                else:
                    [t_data_batch, x_data_batch, y_data_batch, z_data_batch, u_data_batch,
                      v_data_batch, w_data_batch, p_data_batch, c_data_batch, t_eqns_batch, 
                        x_eqns_batch, y_eqns_batch, z_eqns_batch] = map(lambda x:x.unsqueeze(-1),data)
                    txyz_batch =  Variable(torch.concat([t_data_batch, x_data_batch, y_data_batch, z_data_batch],1), requires_grad = True)
                    u_data_batch =  Variable(u_data_batch, requires_grad = True)
                    v_data_batch =  Variable(v_data_batch, requires_grad = True)
                    w_data_batch =  Variable(w_data_batch, requires_grad = True)
            # physics "informed" neural networks
                [c_eqns_pred,
                u_eqns_pred,
                v_eqns_pred,
                w_eqns_pred,
                p_eqns_pred] = torch.split(self.net_cuvwp(txyz_batch),1,1)
                [e1_eqns_pred,
                e2_eqns_pred,
                e3_eqns_pred,
                e4_eqns_pred,
                e5_eqns_pred] = Navier_Stokes_3D(c_eqns_pred,
                                                    u_eqns_pred,
                                                    v_eqns_pred,
                                                    w_eqns_pred,
                                                    p_eqns_pred,
                                                    txyz_batch,
                                                    self.Pec,
                                                    self.Rey)

                if self.other_loss:
                    loss =  mean_squared_error(u_eqns_pred, u_data_batch) + \
                            mean_squared_error(v_eqns_pred, v_data_batch) + \
                            mean_squared_error(w_eqns_pred, w_data_batch) + \
                            mean_squared_error(e2_eqns_pred, 0.0) + \
                            mean_squared_error(e3_eqns_pred, 0.0) + \
                            mean_squared_error(e4_eqns_pred, 0.0) + \
                            mean_squared_error(e5_eqns_pred, 0.0)
                else:
                    loss =  mean_squared_error(c_eqns_pred, c_data_batch) + \
                            mean_squared_error(e1_eqns_pred, 0.0) + \
                            mean_squared_error(e2_eqns_pred, 0.0) + \
                            mean_squared_error(e3_eqns_pred, 0.0) + \
                            mean_squared_error(e4_eqns_pred, 0.0) + \
                            mean_squared_error(e5_eqns_pred, 0.0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                # Print
                if epoch % 10 == 0:
                    elapsed = time.time() - start_time
                    running_time += elapsed/3600.0
                    if rank==0:
                        logger.log(
                                    step=[epoch, i],
                                    data={
                                        "epoch": epoch,
                                        "total_epoch": total_epoch,
                                        "loss": loss,
                                        "elapsed_time": elapsed,
                                        "running_time_hours": running_time,
                                        "learning_rate": self.optimizer.state_dict()['param_groups'][0]['lr'],
                                        "memory_MB": psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 },
                                    verbosity=Verbosity.DEFAULT
                                )
                        
                        sys.stdout.flush()
                    start_time = time.time()
                epoch += 1
        if rank==0:
            torch.save(self.net_cuvwp.state_dict(), self.save_path)
            end_train(model=self, dataset=dataset, T_data=T_data, device=device)

    def predict(self, inputs, device):
        torch.empty_cache()
        self.net_cuvwp.eval()
        [t_test, x_test, y_test, z_test] = map(lambda x: torch.tensor(x, dtype=torch.float32, requires_grad=False,device=device), inputs)
        [c_star,
         u_star,
         v_star,
         w_star,
         p_star] = self.net_cuvwp(t_test, x_test, y_test, z_test)
        
        return c_star, u_star, v_star, w_star, p_star

@end_train_wrapper('npy',type_point='all')
def end_train(model, dataset, T_data, device):
    snap = 100
    [x_test, y_test, z_test, t_test,
    c_test, u_test, v_test, w_test, p_test] = dataset.getTestData(snap, index=0)
    
    # Prediction
    c_pred, u_pred, v_pred, w_pred, p_pred = model.predict((t_test, x_test, y_test, z_test), device)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)
    error_p = relative_error(p_pred - torch.mean(p_pred, dim=0), p_test - np.mean(p_test, axis=0))

    logger.info('Error c: %e' % (error_c))
    logger.info('Error u: %e' % (error_u))
    logger.info('Error v: %e' % (error_v))
    logger.info('Error w: %e' % (error_w))
    logger.info('Error p: %e' % (error_p))

    ################# Save Data ###########################
    
    C_pred = []
    U_pred = []
    V_pred = []
    W_pred = []
    P_pred = []
    for snap in tqdm(range(0, T_data)):
        [x_data, y_data, z_data, t_data, c_data, u_data, v_data, w_data, p_data] = dataset.getTestData(index=snap, snap=T_data)
        # Prediction
        c_pred, u_pred, v_pred, w_pred, p_pred = model.predict([t_data, x_data, y_data, z_data])
  
        C_pred.append(c_pred.detach().cpu().numpy().squeeze())
        U_pred.append(u_pred.detach().cpu().numpy().squeeze())
        V_pred.append(v_pred.detach().cpu().numpy().squeeze())
        W_pred.append(w_pred.detach().cpu().numpy().squeeze())
        P_pred.append(p_pred.detach().cpu().numpy().squeeze())
        # Error
        error_c = relative_error(c_pred, c_data)
        error_u = relative_error(u_pred, u_data)
        error_v = relative_error(v_pred, v_data)
        error_w = relative_error(w_pred, w_data)
        error_p = relative_error(p_pred - torch.mean(p_pred), p_data - np.mean(p_data))
    
        logger.info('Error c: %e' % (error_c))
        logger.info('Error u: %e' % (error_u))
        logger.info('Error v: %e' % (error_v))
        logger.info('Error w: %e' % (error_w))
        logger.info('Error p: %e' % (error_p))
        logger.info("--------------------",snap,"--------------------")
    C_pred = np.array(C_pred).T
    U_pred = np.array(U_pred).T
    V_pred = np.array(V_pred).T
    W_pred = np.array(W_pred).T
    P_pred = np.array(P_pred).T
    scipy.io.savemat('Results/Cylinder3D_results_pread%s.mat' %(time.strftime('%d_%m_%Y')),
                    {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred, 'P_pred':P_pred})
    logger.info('Results already saved!')
    logger.info(f"C_pred shape: {C_pred.shape}, U_pred shape: {U_pred.shape}, "
                f"V_pred shape: {V_pred.shape}, W_pred shape: {W_pred.shape}, "
                f"P_pred shape: {P_pred.shape}")

    dataset.close_f()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=40000, type=int)
    parser.add_argument("--T_data", default=10, type=int)
    parser.add_argument("--N_data",default=100, help='Total number of points', type=int)
    parser.add_argument("--work_size", default=2, type=int)
    parser.add_argument("--path", default="./Datasets_HFM/gen_data_predict_t1.npy", type=str)
    parser.add_argument("--total_epoch", default=10,type=int)
    parser.add_argument("--is_eqns", default=False, help='The flag of auto generate dataset.', type=bool)
    parser.add_argument("--is_other_loss", default=True, help='The flag of using other loss.', type=bool)
    parser.add_argument("--model_path", default=None, help='Loading weights from this path.')
    parser.add_argument("--alpha", default=0.7, help='The weight of Ohter loss ', type=float)
    parser.add_argument("--rey", default=100, type=int)
    parser.add_argument("--pec", default=100, type=int)
    parser.add_argument("--layers", default=10, type=int)
    parser.add_argument("--width", default=200, type=int)
    parser.add_argument("--activation", default='swish', type=str)
    parser.add_argument("--normalization", default=None, type=str)
    parser.add_argument("--model_save_path", default="weights/Cylinder3D.pth", type=str)
    parser.add_argument('--device', required=True, default='cpu', type=str,
                        help='which device to use. cpu, cuda, sdaa optional, cpu default')
    parser.add_argument
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.model_save_path),exist_ok=True)
    batch_size = args.batch_size #总的batchsize，每张卡上batchsize=batch_size/world_size
    T_data = args.T_data #时序长度 all2 1210
    N_data = args.N_data*T_data #总的行数 all2 (63916*1210, 9) 63264 8151 point 1034994
    layers = [4] + args.layers*[args.width] + [5]
    world_size = args.work_size #使用几个GPU训练
    rank = 0 #当前进程排序
    # Load Data
    path = args.path #数据集路径
    total_epoch = args.total_epoch  #训练的epoch,总的迭代次数=total_epoch*N_data/batch_size
    is_eqns = args.is_eqns #是否使用自生成数据
    is_other_loss = args.is_other_loss #是否使用回归损失
    model_path = args.model_path
    alpha = args.alpha #N-S损失占的权重
    Pec = args.pec
    Rey = args.rey
    device = args.device
   # Training
    model = HFM(layers, batch_size,
                Pec = Pec, Rey = Rey, eqns=is_eqns, other_loss=is_other_loss, alpha=alpha,save_path=args.model_save_path)
    
    model.train(0, device, world_size, total_epoch*N_data//batch_size, path, T_data, N_data, model_path, args.activation, args.normalization, args)
