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
import scipy.io
import time
from utilities import neural_net, relative_error
from LoadDataset_pread_ddp import Dataset3D
from tqdm import tqdm
import functools
import argparse
import os
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, "tmp.json"),
    ]
)

def predict(model, inputs, device='sdaa:0'):
    torch.sdaa.empty_cache()
    model.eval()
    [t_test, x_test, y_test, z_test] = map(lambda x: torch.tensor(x, dtype=torch.float32, requires_grad=False,device=device), inputs)
    [c_star,
        u_star,
        v_star,
        w_star,
        p_star] = torch.split(model(torch.concat((t_test, x_test, y_test, z_test),1)),1,1)
    return c_star, u_star, v_star, w_star, p_star

def predict2D(model, inputs, device='sdaa:0'):
    torch.sdaa.empty_cache()
    model.eval()
    [t_test, x_test, y_test] = map(lambda x: torch.tensor(x, dtype=torch.float32, requires_grad=False,device=device), inputs)
    [c_star,
        u_star,
        v_star,
        p_star] = model(t_test, x_test, y_test)
    return c_star, u_star, v_star, p_star

def predict_save2mat(model, dataset, T_data, device='sdaa:0'):
    model_predict = functools.partial(predict, model=model)
    snap = 100
    [x_test, y_test, z_test, t_test,
    c_test, u_test, v_test, w_test, p_test] = dataset.getTestData(snap=T_data, index=snap)
    
    # Prediction
    c_pred, u_pred, v_pred, w_pred, p_pred = model_predict(inputs=(t_test, x_test, y_test, z_test), device=device)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)
    error_p = relative_error(p_pred - torch.mean(p_pred, dim=0), p_test - np.mean(p_test, axis=0))

    logger.info(
        f"Error c: {error_c:.6e}, "
        f"Error u: {error_u:.6e}, "
        f"Error v: {error_v:.6e}, "
        f"Error w: {error_w:.6e}, "
        f"Error p: {error_p:.6e}"
    )

    ################# Save Data ###########################
    
    C_pred = []
    U_pred = []
    V_pred = []
    W_pred = []
    P_pred = []
    for snap in tqdm(range(0, T_data)):
        [x_data, y_data, z_data, t_data, c_data, u_data, v_data, w_data, p_data] = dataset.getTestData(index=snap, snap=T_data)
        # Prediction
        c_pred, u_pred, v_pred, w_pred, p_pred = model_predict(inputs=[t_data, x_data, y_data, z_data])
  
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
    
        logger.info(
            f"Error c: {error_c:.6e}, "
            f"Error u: {error_u:.6e}, "
            f"Error v: {error_v:.6e}, "
            f"Error w: {error_w:.6e}, "
            f"Error p: {error_p:.6e}"
        )
        logger.info(f"--------------------{snap}--------------------")

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
    # dist.barrier()
    dataset.close_f()


def predict_save2npy(model, dataset, T_data, device='sdaa:0',type_point='all'):
    model_predict = functools.partial(predict,model=model)
    model.eval()
    snap = 100
    [x_test, y_test, z_test, t_test,
    c_test, u_test, v_test, w_test, p_test] = dataset.getTestData(snap=T_data, index=snap)
    
    # Prediction
    c_pred, u_pred, v_pred, w_pred, p_pred = model_predict(inputs=(t_test, x_test, y_test, z_test), device=device)
    
    # Error
    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_w = relative_error(w_pred, w_test)
    error_p = relative_error(p_pred - torch.mean(p_pred, dim=0), p_test - np.mean(p_test, axis=0))

    logger.info(
        f"Error c: {error_c:.6e}, "
        f"Error u: {error_u:.6e}, "
        f"Error v: {error_v:.6e}, "
        f"Error w: {error_w:.6e}, "
        f"Error p: {error_p:.6e}"
    )

    ################# Save Data ###########################
    save_path = f"Results/Cylinder3D_results_pread{(time.strftime('%H_%d_%m_%Y'))}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for snap in tqdm(range(0, T_data)):
            [x_data, y_data, z_data, t_data, c_data, u_data, v_data, w_data, p_data] = dataset.getTestData(index=snap, snap=T_data)
            # Prediction
            c_pred, u_pred, v_pred, w_pred, p_pred = model_predict(inputs=(t_data, x_data, y_data, z_data), device=device)
            res = [u_pred, v_pred, w_pred, p_pred, c_pred]
            res = list(map(lambda x : x.detach().cpu().numpy(), res))
            if type_point=='all':
                res = [t_data, x_data, y_data, z_data] + res
            data = np.concatenate(res, axis=1)
            # print(data.shape)
            # Error
            error_c = relative_error(c_pred, c_data)
            error_u = relative_error(u_pred, u_data)
            error_v = relative_error(v_pred, v_data)
            error_w = relative_error(w_pred, w_data)
            error_p = relative_error(p_pred - torch.mean(p_pred), p_data - np.mean(p_data))
        
            logger.info(
                f"Error c: {error_c:.6e}, "
                f"Error u: {error_u:.6e}, "
                f"Error v: {error_v:.6e}, "
                f"Error w: {error_w:.6e}, "
                f"Error p: {error_p:.6e}"
            )
            logger.info(f"--------------------{snap}--------------------")
            np.save(f, data)
        logger.info(f'Results already saved! Save: {save_path}')
    dataset.close_f()

def predict_2D_npy(model, dataset, T_data, device='sdaa:0'):
    model_predict = functools.partial(predict2D,model=model)
    model.eval()
    snap = 100

    ################# Save Data ###########################
    save_path = f"Results/Cylinder2D_results_pread{(time.strftime('%H_%d_%m_%Y'))}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for snap in tqdm(range(0, T_data)):
            [x_data, y_data, z_data, t_data, c_data, u_data, v_data, w_data, p_data] = dataset.getTestData(index=snap, snap=T_data)
            # Prediction
            c_pred, u_pred, v_pred, p_pred = model_predict(inputs=(t_data, x_data, y_data), device=device)
            res = [u_pred, v_pred, p_pred, c_pred]
            res = list(map(lambda x : x.detach().cpu().numpy(), res))
            res = [t_data, x_data, y_data] + res
            data = np.concatenate(res, axis=1)
            # print(data.shape)
            # Error
            error_c = relative_error(c_pred, c_data)
            error_u = relative_error(u_pred, u_data)
            error_v = relative_error(v_pred, v_data)
            error_p = relative_error(p_pred - torch.mean(p_pred), p_data - np.mean(p_data))
        
            logger.info(
                f"Error c: {error_c:.6e}, "
                f"Error u: {error_u:.6e}, "
                f"Error v: {error_v:.6e}, "
                f"Error p: {error_p:.6e}"
            )
            logger.info(f"--------------------{snap}--------------------")
            np.save(f, data)
        logger.info(f'Results already saved! Save: {save_path}')
    dataset.close_f()


def predict_save2npy_big(model, dataset, T_data, N_data, device='sdaa:0',type_point='all',batch_size=400000):
    model_predict = functools.partial(predict,model=model)
    model.eval()
    N_data = N_data/T_data
    mod = 1 if N_data%batch_size > 0 else 0
    ################# Save Data ###########################
    save_path = f"Results/Cylinder3D_results_pread{(time.strftime('%H_%d_%m_%Y'))}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        for snap in tqdm(range(0, T_data)):
            batch_data=[]
            for idx in range(int(N_data//batch_size)+mod):
                end = int(min(snap+(idx+1)*(T_data*batch_size),N_data*T_data))
                start = int(snap+idx*(T_data*batch_size))
                [x_data, y_data, z_data, t_data, c_data, u_data, v_data, w_data, p_data] = dataset.getTestData(index=start,end=end, snap=T_data)
                # Prediction
                c_pred, u_pred, v_pred, w_pred, p_pred = model_predict(inputs=(t_data, x_data, y_data, z_data), device=device)
                res = [u_pred, v_pred, w_pred, p_pred, c_pred]
                res = list(map(lambda x : x.detach().cpu().numpy(), res))
                res[2] = w_data
                if type_point=='all':
                    res = [t_data, x_data, y_data, z_data] + res
                data = np.concatenate(res, axis=1)
                # print(data.shape)
                # Error
                error_c = relative_error(c_pred, c_data)
                error_u = relative_error(u_pred, u_data)
                error_v = relative_error(v_pred, v_data)
                error_w = relative_error(w_pred, w_data)
                error_p = relative_error(p_pred - torch.mean(p_pred), p_data - np.mean(p_data))
                if idx==0:
                    logger.info(
                        f"Error c: {error_c:.6e}, "
                        f"Error u: {error_u:.6e}, "
                        f"Error v: {error_v:.6e}, "
                        f"Error w: {error_w:.6e}, "
                        f"Error p: {error_p:.6e}"
                    )
                    logger.info(f"--------------------{snap}--------------------")
                batch_data.append(data)
            batch_data = np.vstack(batch_data)
            assert len(batch_data)==N_data,len(batch_data)
            np.save(f, batch_data)
        logger.info(f'Results already saved! Save: {save_path}')
        dataset.close_f()

def get_model(layers, model_path='weights/Cylinder3D.pth',device='sdaa:0',normlization='BN',activation='swish'):
    net_cuvwp = neural_net(layers = layers, device=device,normalization=normlization, activation=activation)
    net_cuvwp.load_state_dict(torch.load(model_path))
    net_cuvwp.to(device)
    return net_cuvwp

def get_dataset(path, T_data, N_data, eqns=False, other_loss=False):
    dataset = Dataset3D(path=path,T_data=T_data,N_data=N_data, eqns=eqns, other_loss=other_loss)
    return dataset

def end_train_wrapper(save_type='npy',type_point='all',model_type='3D'):
    def wrapper(fun):
        def predict(*args, **kwargs):
            model = kwargs['model'].net_cuvwp
            dataset =  kwargs['dataset']
            T_data =  kwargs['T_data']
            device =  kwargs['device']
            if model_type=='3D':
                if save_type=='mat':
                    predict_save2mat(model, dataset, T_data, device)
                elif save_type == 'npy':
                    if int(dataset.N_data//T_data)>400000:
                        predict_save2npy_big(model=model,dataset=dataset,device=device,T_data=T_data, N_data=dataset.N_data, type_point=type_point, batch_size=400000)
                    else:    
                        predict_save2npy(model, dataset, T_data, device,type_point=type_point)
                else:
                    raise NotImplementedError
            elif model_type =='2D':
                predict_2D_npy(model, dataset, T_data, device)
            return True
        return predict
    return wrapper

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--T_data', type=int, default=201)
    parser.add_argument('--N_data', type=int, default=63264)
    parser.add_argument('--layers',type=int, default=10)
    parser.add_argument('--width',type=int,default=200)
    parser.add_argument('--normlization', default=None)
    parser.add_argument('--activation', default='swish')
    parser.add_argument('--batch_size', type=int, default=400000)
    parser.add_argument('--model_type', default='3D')
    args = parser.parse_args()
    T_data = args.T_data
    N_data = args.T_data*args.N_data #总的行数 all2 (63916*1210, 9) 63264
    if args.model_type == '3D':
        layers = [4] + args.layers*[args.width] + [5]
    else:
        layers = [3] + args.layers*[args.width] + [4]
    # Load Data
    # path = '/data/application/common/Datasets_HFM/all2_dataset_t201_c_norm.npy'
    device = torch.device("sdaa:0")#确定使用的gpu序号
    is_eqns = False #是否使用自生成数据
    is_other_loss = False #是否使用回归损失
    # model_path = 'weights/Cylinder3D.pth'
    path = args.data_path
    model_path = args.model_path
    model = get_model(layers, model_path, device, normlization=args.normlization, activation=args.activation)
    dataset = get_dataset(path, T_data, N_data, is_eqns, is_other_loss)
    if args.model_type=='3D':
        if args.N_data>400000:
            predict_save2npy_big(model=model, dataset=dataset, T_data=T_data,N_data=N_data, device=device,batch_size=args.batch_size)
        else:
            predict_save2npy(model, dataset, T_data, device)
    else:
        predict_2D_npy(model, dataset, T_data, device)