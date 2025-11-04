import datetime
import itertools
import os
import time
import sys
import torch
import torch_sdaa
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.cyclegan import discriminator, generator
from utils.callbacks import LossHistory
from utils.dataloader import CycleGan_dataset_collate, CycleGanDataset
from utils.utils import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit import fit_one_epoch
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['MASTER_ADDR'] = '127.0.0.1' # 设置IP
# 从外部获取local_rank参数
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if __name__ == "__main__":
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Sdaa            = True
    #---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    #---------------------------------------------------------------------#
    distributed     = True
    #---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    #---------------------------------------------------------------------#
    fp16            = False
    #--------------------------------------------------------------------------#
    #   如果想要断点续练就将model_path设置成logs文件夹下已经训练的权值文件。 
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的。
    #   如果想要让模型从0开始训练，则设置model_path = ''。
    #--------------------------------------------------------------------------#
    G_model_A2B_path    = ""
    G_model_B2A_path    = ""
    D_model_A_path      = ""
    D_model_B_path      = ""
    #-------------------------------#
    #   输入图像大小的设置
    #-------------------------------#
    input_shape     = [256, 256]
    
    #------------------------------#
    #   训练参数设置
    #------------------------------#
    Init_Epoch      = 0
    Epoch           = 200
    batch_size      = 4
    
    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 2e-4
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=2e-4
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.5
    weight_decay        = 0
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0  
    #------------------------------------------------------------------#
    num_workers         = 4
    #------------------------------#
    #   每隔50个step保存一次图片
    #------------------------------#
    photo_save_step     = 50
    
    #------------------------------------------#
    #   获得图片路径
    #------------------------------------------#
    annotation_path = "/datasets/train_lines.txt"

    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.sdaa.device_count()
    if distributed:
        #更改
        '''dist.init_process_group(backend="tccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("sdaa", local_rank)'''
        #更改
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
        '''if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)'''
    else:
        device          = torch.device('sdaa' if torch.sdaa.is_available() else 'cpu')
        local_rank      = 0
    #------------------------------------------#
    #   生成网络和评价网络
    #------------------------------------------#
    G_model_A2B = generator()
    G_model_B2A = generator()
    D_model_A   = discriminator()
    D_model_B   = discriminator()
    
    model_train_log_path = "scripts/train_sdaa_3rd.log"
    model_train_log_file = open(model_train_log_path, 'w')
    #------------------------------------------#
    #   将训练好的模型重新载入
    #------------------------------------------#
    if G_model_A2B_path != '':
        pretrained_dict = torch.load(G_model_A2B_path, map_location=device)
        G_model_A2B.load_state_dict(pretrained_dict)
    if G_model_B2A_path != '':
        pretrained_dict = torch.load(G_model_B2A_path, map_location=device)
        G_model_B2A.load_state_dict(pretrained_dict)
    if D_model_A_path != '':
        pretrained_dict = torch.load(D_model_A_path, map_location=device)
        D_model_A.load_state_dict(pretrained_dict)
    if D_model_B_path != '':
        pretrained_dict = torch.load(D_model_B_path, map_location=device)
        D_model_B.load_state_dict(pretrained_dict)
    
    #----------------------#
    #   获得损失函数
    #----------------------#
    BCE_loss = nn.BCEWithLogitsLoss()
    MSE_loss = nn.MSELoss()
    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, [G_model_A2B, G_model_B2A, D_model_A, D_model_B], input_shape=input_shape)
    else:
        loss_history    = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        
        scaler = torch.amp.GradScaler('sdaa',)
    else:
        scaler = None

    G_model_A2B_train = G_model_A2B.train()
    G_model_B2A_train = G_model_B2A.train()
    D_model_A_train = D_model_A.train()
    D_model_B_train = D_model_B.train()
    
    if Sdaa:
        if distributed:
            #print("+++++++++=============")
            G_model_A2B_train = G_model_A2B_train.sdaa(local_rank)
            G_model_A2B_train = DDP(G_model_A2B_train, device_ids=[local_rank], find_unused_parameters=True)
            
            G_model_B2A_train = G_model_B2A_train.sdaa(local_rank)
            G_model_B2A_train = DDP(G_model_B2A_train, device_ids=[local_rank], find_unused_parameters=True)
            
            D_model_A_train = D_model_A_train.sdaa(local_rank)
            D_model_A_train = DDP(D_model_A_train, device_ids=[local_rank], find_unused_parameters=True)
            
            D_model_B_train = D_model_B_train.sdaa(local_rank)
            D_model_B_train = DDP(D_model_B_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            cudnn.benchmark = True
            
            G_model_A2B_train = torch.nn.DataParallel(G_model_A2B_train)
            G_model_A2B_train = G_model_A2B_train.sdaa()

            G_model_B2A_train = torch.nn.DataParallel(G_model_B2A_train)
            G_model_B2A_train = G_model_B2A_train.sdaa()

            D_model_A_train = torch.nn.DataParallel(D_model_A_train)
            D_model_A_train = D_model_A_train.sdaa()

            D_model_B_train = torch.nn.DataParallel(D_model_B_train)
            D_model_B_train = D_model_B_train.sdaa()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(annotation_path) as f:
        lines = f.readlines()
    annotation_lines_A, annotation_lines_B = [], []
    for annotation_line in lines:
        annotation_lines_A.append(annotation_line) if int(annotation_line.split(';')[0]) == 0 else annotation_lines_B.append(annotation_line)
    num_train = max(len(annotation_lines_A), len(annotation_lines_B))

    if local_rank == 0:
        show_config(
            input_shape = input_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
            )

    #------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Epoch总训练世代
    #------------------------------------------------------#
    if True:
        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        G_optimizer = {
            'adam'  : optim.Adam(itertools.chain(G_model_A2B_train.parameters(), G_model_B2A_train.parameters()), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(itertools.chain(G_model_A2B_train.parameters(), G_model_B2A_train.parameters()), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        D_optimizer_A = {
            'adam'  : optim.Adam(D_model_A_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(D_model_A_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        D_optimizer_B = {
            'adam'  : optim.Adam(D_model_B_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(D_model_B_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        train_dataset   = CycleGanDataset(annotation_lines_A, annotation_lines_B, input_shape)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
    
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=CycleGan_dataset_collate, sampler=train_sampler)

        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        start_time = time.time()
        
        for epoch in range(Init_Epoch, Epoch):
            time0 = time.time()
            if time0 - start_time > 3600*3:
                sys.exit(1)
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(G_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(D_optimizer_A, lr_scheduler_func, epoch)
            set_optimizer_lr(D_optimizer_B, lr_scheduler_func, epoch)
            
            fit_one_epoch(G_model_A2B_train, G_model_B2A_train, D_model_A_train, D_model_B_train, G_model_A2B, G_model_B2A, D_model_A, D_model_B, loss_history,
                        G_optimizer, D_optimizer_A, D_optimizer_B, BCE_loss, MSE_loss, epoch, epoch_step, gen, Epoch, Sdaa, fp16, scaler, save_period, save_dir, photo_save_step, model_train_log_file,local_rank)

            if distributed:
                dist.barrier()
        if local_rank ==  0:
            model_train_log_file.close()