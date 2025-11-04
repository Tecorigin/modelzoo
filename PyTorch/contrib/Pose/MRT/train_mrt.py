# Adapted to tecorigin hardware
import torch
import torch_sdaa
import torch.optim as optim
import torch_dct as dct #https://github.com/zh217/torch-dct
import time
import os
from MRT.Models import Transformer,Discriminator
from utils import disc_l2_loss,adv_disc_l2_loss
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import argparse
from data import DATA
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

scaler = torch_sdaa.amp.GradScaler()
os.makedirs('saved_model',exist_ok=True)
logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'saved_model/log.json'),
    ]
)

# parser
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', required=False, default=20,
                    type=int, help='number of total epochs to run (default: EPOCH in the cfg configuration)')
parser.add_argument('--batch_size','--bs' ,required=False, default=64,
                    type=int, help='mini-batch size per device (default: BATCH_SIZE in the cfg configuration )')
parser.add_argument('--device', required=False, default='sdaa', type=str,
                    help="The device for this task, e.g. sdaa or cpu, default: sdaa")
parser.add_argument('--data_path', required=False, default='./mocap', type=str,
                    help="The root path of mocap dataset, default: ./mocap")
parser.add_argument("--autocast", default=False, action='store_true', help="open autocast for amp")
parser.add_argument('--ddp', default=False, action='store_true', help="whether to use ddp")
args = parser.parse_args()
logger.info(args)

dataset = DATA(data_path = args.data_path)
batch_size=args.batch_size

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

from discriminator_data import D_DATA
real_=D_DATA(data_path = args.data_path)

real_motion_dataloader = torch.utils.data.DataLoader(real_, batch_size=batch_size, shuffle=True)
real_motion_all=list(enumerate(real_motion_dataloader))

device=args.device


lrate=0.0003
lrate2=0.0005


def train(rank, world_size):
    if isinstance(rank, str) and ('sdaa' in rank or 'cpu' in rank):
        device = rank
        torch.sdaa.set_device(device)
        rank = 0
    elif isinstance(rank, int) and args.ddp:
        device = f"sdaa:{rank}"
        torch.sdaa.set_device(device)
    else:
        raise ValueError(f"rank should be 'sdaa' or 'cpu' or an integer when using ddp, got {rank}")
    # model
    model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
                n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)

    discriminator = Discriminator(d_word_vec=45, d_model=45, d_inner=256,
                n_layers=3, n_head=8, d_k=32, d_v=32,device=device).to(device)
    # ddp
    if args.ddp:
        dist.init_process_group('tccl', rank=rank, world_size=world_size)
        model = DDP(model, find_unused_parameters=True)
        discriminator = DDP(discriminator, find_unused_parameters=True)
    # params
    params = [
        {"params": model.parameters(), "lr": lrate}
    ]
    optimizer = optim.Adam(params)
    params_d = [
        {"params": discriminator.parameters(), "lr": lrate}
    ]
    optimizer_d = optim.Adam(params_d)
    # train
    for epoch in range(args.epoch):
        total_loss=0
        
        for j,data in enumerate(dataloader,0):
            if rank == 0: print(f"epoch: {epoch}/{args.epoch}, overall: {100*(epoch/args.epoch + j/(args.epoch*dataloader.__len__())):.1f}%")
            time1 = time.time()
            use=None
            input_seq,output_seq=data
            input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device) # batch, N_person, 15 (15 fps 1 second), 45 (15joints xyz) 
            output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device) # batch, N_persons, 46 (last frame of input + future 3 seconds), 45 (15joints xyz) 
            
            # first 1 second predict future 1 second
            input_=input_seq.view(-1,15,input_seq.shape[-1]) # batch x n_person ,15: 15 fps, 1 second, 45: 15joints x 3
            
            output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

            input_ = dct.dct(input_)
            with torch_sdaa.amp.autocast(enabled=args.autocast):
                rec_=model.forward(input_[:,1:15,:]-input_[:,:14,:],dct.idct(input_[:,-1:,:]),input_seq,use).to(torch.float32)

                rec=dct.idct(rec_)

                # first 2 seconds predict 1 second
                new_input=torch.cat([input_[:,1:15,:]-input_[:,:14,:],dct.dct(rec_)],dim=-2)
                
                new_input_seq=torch.cat([input_seq,output_seq[:,:,1:16]],dim=-2)
                new_input_=dct.dct(new_input_seq.reshape(-1,30,45))
                new_rec_=model.forward(new_input_[:,1:,:]-new_input_[:,:29,:],dct.idct(new_input_[:,-1:,:]),new_input_seq,use)

                new_rec=dct.idct(new_rec_)

                # first 3 seconds predict 1 second
                new_new_input_seq=torch.cat([input_seq,output_seq[:,:,1:31]],dim=-2)
                new_new_input_=dct.dct(new_new_input_seq.reshape(-1,45,45))
                new_new_rec_=model.forward(new_new_input_[:,1:,:]-new_new_input_[:,:44,:],dct.idct(new_new_input_[:,-1:,:]),new_new_input_seq,use)

                new_new_rec=dct.idct(new_new_rec_)
                
                rec=torch.cat([rec,new_rec,new_new_rec],dim=-2)
                
                results=output_[:,:1,:]
                for i in range(1,31+15):
                    results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
                results=results[:,1:,:]
            
                loss=torch.mean((rec[:,:,:]-(output_[:,1:46,:]-output_[:,:45,:]))**2)
            
            
            if (j+1)%2==0:
                
                fake_motion=results

                with torch_sdaa.amp.autocast(enabled=args.autocast):
                    disc_loss=disc_l2_loss(discriminator(fake_motion))
                loss=loss+0.0005*disc_loss
                
                fake_motion=fake_motion.detach()

                real_motion=real_motion_all[int(j/2)][1][1]
                real_motion=real_motion.view(-1,46,45)[:,1:46,:].float().to(device)
                with torch_sdaa.amp.autocast(enabled=args.autocast):
                    fake_disc_value = discriminator(fake_motion)
                    real_disc_value = discriminator(real_motion)

                    d_motion_disc_real, d_motion_disc_fake, d_motion_disc_loss = adv_disc_l2_loss(real_disc_value, fake_disc_value)
                
                optimizer_d.zero_grad()
                if args.autocast:
                    scaler.scale(d_motion_disc_loss).backward()
                    scaler.step(optimizer_d)  
                    scaler.update()  
                else:
                    d_motion_disc_loss.backward()
                    optimizer_d.step()
            
        
            optimizer.zero_grad()
            if args.autocast:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()  
            else:
                loss.backward()
                optimizer.step()
            if rank == 0:
                print(f"loss: {loss.item()}", end=';')
                print(f"time: {time.time()-time1}")
    
    
    
            total_loss=total_loss+loss
            time_cost = time.time()-time1
            log_data = {'train.rank': rank,
                        'train.loss': loss.item(),
                        'train.time': time_cost,
                        'train.ips': batch_size/(time_cost)}
            logger.log(step = (epoch, j), data = log_data, verbosity=Verbosity.DEFAULT)
            
            
        if rank == 0: print('epoch:',epoch,'loss:',total_loss/(j+1))
        if (epoch+1)%5==0:
            save_path=f'./saved_model/{epoch}.model'
            torch.save(model.state_dict(),save_path)
    
    if rank == 0:    
        save_path=f'./saved_model/{epoch}.model'
        torch.save(model.state_dict(),save_path)
# ddp
def main():
    if args.ddp:
        os.environ['MASTER_ADDR'] = 'localhost' # 设置IP
        os.environ['MASTER_PORT'] = '29503'     # 设置端口号
        world_size=torch.sdaa.device_count()
        mp.spawn(train,
            args=(world_size,),
            nprocs=world_size,
            join=True)
    else:
        train(args.device,1)

if __name__ == '__main__':
    main()
        
        
