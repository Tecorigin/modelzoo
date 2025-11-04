# Adapted to tecorigin hardware
# https://github.com/jiashunwang/MRT/pull/2
from typing import OrderedDict
import torch
import torch_sdaa
import numpy as np
import torch_dct as dct
from MRT.Models import Transformer
import matplotlib.pyplot as plt
import numpy as np
from data import TESTDATA
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
import argparse

logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'saved_model/val.json'),
    ]
)


parser = argparse.ArgumentParser()
parser.add_argument('--device', required=False, default='sdaa', type=str,
                    help="The device for this task, e.g. sdaa or cpu, default: sdaa")
parser.add_argument('--data_path', required=False, default='./mocap', type=str,
                    help="The root path of mocap dataset, default: ./mocap")
parser.add_argument('--checkpoint', required=False, default='saved_model/19.model', type=str, help='checkpoint file path')
args = parser.parse_args()
logger.info(args)

dataset_name='mocap'
test_dataset = TESTDATA(dataset=dataset_name, data_path = args.data_path)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

device=args.device



model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,
            n_layers=3, n_head=8, d_k=64, d_v=64,device=device).to(device)




plot=False
gt=False

state_dict = torch.load(args.checkpoint, map_location=device)
# ddp weight loading
if 'module' in list(state_dict.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    state_dict = new_state_dict
model.load_state_dict(state_dict) 


body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)


losses=[]

total_loss=0
loss_list1=[]
loss_list2=[]
loss_list3=[]
with torch.no_grad():
    model.eval()
    loss_list=[]
    for jjj,data in enumerate(test_dataloader,0):
        print(jjj)
        #if jjj!=20:
        #    continue
        input_seq,output_seq=data
        
        input_seq=torch.tensor(input_seq,dtype=torch.float32).to(device)
        output_seq=torch.tensor(output_seq,dtype=torch.float32).to(device)
        n_joints=int(input_seq.shape[-1]/3)
        use=[input_seq.shape[1]]
        
        input_=input_seq.view(-1,15,input_seq.shape[-1])
  
    
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1],-1,input_seq.shape[-1])

        input_ = dct.dct(input_)
        output__ = dct.dct(output_[:,:,:])
        
        
        rec_=model.forward(input_[:,1:15,:]-input_[:,:14,:],dct.idct(input_[:,-1:,:]),input_seq,use)
        
        rec=dct.idct(rec_)

        results=output_[:,:1,:]
        for i in range(1,16):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]

        new_input_seq=torch.cat([input_seq,results.reshape(input_seq.shape)],dim=-2)
        new_input=dct.dct(new_input_seq.reshape(-1,30,45))
        
        new_rec_=model.forward(new_input[:,1:,:]-new_input[:,:-1,:],dct.idct(new_input[:,-1:,:]),new_input_seq,use)
        
        
        new_rec=dct.idct(new_rec_)
        
        new_results=new_input_seq.reshape(-1,30,45)[:,-1:,:]
        for i in range(1,16):
            new_results=torch.cat([new_results,new_input_seq.reshape(-1,30,45)[:,-1:,:]+torch.sum(new_rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        new_results=new_results[:,1:,:]
        
        results=torch.cat([results,new_results],dim=-2)

        rec=torch.cat([rec,new_rec],dim=-2)

        results=output_[:,:1,:]

        for i in range(1,16+15):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]

        new_new_input_seq=torch.cat([input_seq,results.unsqueeze(0)],dim=-2)
        new_new_input=dct.dct(new_new_input_seq.reshape(-1,45,45))
        
        new_new_rec_=model.forward(new_new_input[:,1:,:]-new_new_input[:,:-1,:],dct.idct(new_new_input[:,-1:,:]),new_new_input_seq,use)


        new_new_rec=dct.idct(new_new_rec_)
        rec=torch.cat([rec,new_new_rec],dim=-2)

        results=output_[:,:1,:]

        for i in range(1,31+15):
            results=torch.cat([results,output_[:,:1,:]+torch.sum(rec[:,:i,:],dim=1,keepdim=True)],dim=1)
        results=results[:,1:,:]
        
        prediction_1=results[:,:15,:].view(results.shape[0],-1,n_joints,3)
        prediction_2=results[:,:30,:].view(results.shape[0],-1,n_joints,3)
        prediction_3=results[:,:45,:].view(results.shape[0],-1,n_joints,3)

        gt_1=output_seq[0][:,1:16,:].view(results.shape[0],-1,n_joints,3)
        gt_2=output_seq[0][:,1:31,:].view(results.shape[0],-1,n_joints,3)
        gt_3=output_seq[0][:,1:46,:].view(results.shape[0],-1,n_joints,3)

        if dataset_name=='mocap':
            #match the scale with the paper, also see line 63 in mix_mocap.py
            loss1=torch.sqrt(((prediction_1/1.8 - gt_1/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            loss2=torch.sqrt(((prediction_2/1.8 - gt_2/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            loss3=torch.sqrt(((prediction_3/1.8 - gt_3/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()

            #pose with align
            # loss1=torch.sqrt((((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            # loss2=torch.sqrt((((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            # loss3=torch.sqrt((((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:])/1.8) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()


        if dataset_name=='mupots':
            loss1=torch.sqrt(((prediction_1 - gt_1) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            loss2=torch.sqrt(((prediction_2 - gt_2) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            loss3=torch.sqrt(((prediction_3 - gt_3) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            
            #pose with align
            # loss1=torch.sqrt(((prediction_1 - prediction_1[:,:,0:1,:] - gt_1 + gt_1[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            # loss2=torch.sqrt(((prediction_2 - prediction_2[:,:,0:1,:] - gt_2 + gt_2[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()
            # loss3=torch.sqrt(((prediction_3 - prediction_3[:,:,0:1,:] - gt_3 + gt_3[:,:,0:1,:]) ** 2).sum(dim=-1)).mean(dim=-1).mean(dim=-1).cpu().numpy().tolist()

        
        loss_list1.append(np.mean(loss1))#+loss1
        loss_list2.append(np.mean(loss2))#+loss2
        loss_list3.append(np.mean(loss3))#+loss3
        
        loss=torch.mean((rec[:,:,:]-(output_[:,1:46,:]-output_[:,:45,:]))**2)
        losses.append(loss)
        

        rec=results[:,:,:]
        
        rec=rec.reshape(results.shape[0],-1,n_joints,3)
        
        input_seq=input_seq.view(results.shape[0],15,n_joints,3)
        pred=torch.cat([input_seq,rec],dim=1)
        output_seq=output_seq.view(results.shape[0],-1,n_joints,3)[:,1:,:,:]
        all_seq=torch.cat([input_seq,output_seq],dim=1)
        

        pred=pred[:,:,:,:].cpu()
        all_seq=all_seq[:,:,:,:].cpu()
        
        
        if plot:
            fig = plt.figure(figsize=(10, 4.5))
            fig.tight_layout()
            ax = fig.add_subplot(111, projection='3d')
            
            plt.ion()
            
            length=45+15
            length_=45+15
            i=0

            p_x=np.linspace(-10,10,15)
            p_y=np.linspace(-10,10,15)
            X,Y=np.meshgrid(p_x,p_y)
            
            
            while i < length_:
                
                ax.lines = []

                for x_i in range(p_x.shape[0]):
                    temp_x=[p_x[x_i],p_x[x_i]]
                    temp_y=[p_y[0],p_y[-1]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

                for y_i in range(p_x.shape[0]):
                    temp_x=[p_x[0],p_x[-1]]
                    temp_y=[p_y[y_i],p_y[y_i]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)

                for j in range(results.shape[0]):
                    
                    xs=pred[j,i,:,0].cpu().numpy()
                    ys=pred[j,i,:,1].cpu().numpy()
                    zs=pred[j,i,:,2].cpu().numpy()
                    
                    alpha=1
                    ax.plot( zs,xs, ys, 'y.',alpha=alpha)
                    
                    if gt:
                        x=all_seq[j,i,:,0].cpu().numpy()
                        
                        y=all_seq[j,i,:,1].cpu().numpy()
                        z=all_seq[j,i,:,2].cpu().numpy()
                    
                    
                        ax.plot( z,x, y, 'y.')


                    plot_edge=True
                    if plot_edge:
                        for edge in body_edges:
                            x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                            y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                            z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                            if i>=15:
                                ax.plot(z,x, y, zdir='z',c='blue',alpha=alpha)
                                
                            else:
                                ax.plot(z,x, y, zdir='z',c='green',alpha=alpha)
                            
                            if gt:
                                x=[all_seq[j,i,edge[0],0],all_seq[j,i,edge[1],0]]
                                y=[all_seq[j,i,edge[0],1],all_seq[j,i,edge[1],1]]
                                z=[all_seq[j,i,edge[0],2],all_seq[j,i,edge[1],2]]
                            
                                if i>=15:
                                    ax.plot( z,x, y, 'yellow',alpha=0.8)
                                else:
                                    ax.plot( z, x, y, 'green')
                            
                   
                    ax.set_xlim3d([-3 , 3])
                    ax.set_ylim3d([-3 , 3])
                    ax.set_zlim3d([0,3])
                    # ax.set_xlim3d([-8 , 8])
                    # ax.set_ylim3d([-8 , 8])
                    # ax.set_zlim3d([0,5])
                    # ax.set_xticklabels([])
                    # ax.set_yticklabels([])
                    # ax.set_zticklabels([])
                    ax.set_axis_off()
                    #ax.patch.set_alpha(1)
                    #ax.set_aspect('equal')
                    #ax.set_xlabel("x")
                    #ax.set_ylabel("y")
                    #ax.set_zlabel("z")
                    plt.title(str(i),y=-0.1)
                plt.pause(0.1)
                i += 1

            
            plt.ioff()
            plt.savefig('1.png')
            import pdb;pdb.set_trace()
            plt.close()

            


    # As mentioned in http://mocap.cs.cmu.edu/info.php,
    # the "The File Formats - asf/amc" part
    # to convert the unit in CMU-Mocap to meters
    scale=(1.0/0.45)*2.54/100.0
    to_decimeter = 10

    val_res = {
        '1s': np.mean(loss_list1) * scale * to_decimeter,
        '2s': np.mean(loss_list2) * scale * to_decimeter,
        '3s': np.mean(loss_list3) * scale * to_decimeter,
    }
    print('avg 1 second',val_res['1s'])
    print('avg 2 seconds',val_res['2s'])
    print('avg 3 seconds',val_res['3s'])
    logger.info(val_res)
    
    
