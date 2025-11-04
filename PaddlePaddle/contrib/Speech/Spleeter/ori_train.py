import os
import warnings
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.resolve()
os.environ['PYTHONPATH'] = str(root_dir)
sys.path.insert(0, str(root_dir))

from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'json_log.log'),
]
)
import psutil
import paddle
import math
import time
from tqdm import tqdm
import json
from dataset_v2 import TrainDataset

from model import UNet, MultiLoss
from paddle.io import DataLoader
from visualdl import LogWriter

import logging
import paddle.nn as nn
import paddle.audio as paddleaudio
import soundfile as sf
import paddle.distributed as dist
from paddle import DataParallel as DDP
from util import HParams
from util import deDaoZuiXinMoXing
from separator import Separator
import warnings
warnings.filterwarnings('ignore')

device = "sdaa"
paddle.set_device(device)
#paddleaudio.backends.set_backend('soundfile')
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout,level=logging.INFO)
logger = logging.getLogger('training Spleeter')
f = logging.FileHandler('train.log',encoding="utf-8")
f.setLevel(logging.INFO)
logger.addHandler(f)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

step = 0
dist.init_parallel_env()
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

def get_data(path, drop_last=False):
    batch_size:int = hps['batch_size'] # 批次大小
    data_buffer1:list = [] #缓冲区
    data_buffer2:list = [] #缓冲区
    data_buffer3:list = [] #缓冲区
    count = 0 # 计数器
    # 先过滤，只留下真的数据集
    l = [i for i in os.listdir(path) if (i.endswith('.pdtensor') and i.startswith('data_'))]
    # 卡的数量
    world_size = paddle.distributed.get_world_size()
    # 每一张卡的数据集长度
    length = len(l) // world_size
    # 每一张卡的数据集
    rank = paddle.distributed.get_rank()
    l = l[rank * length : (rank + 1) * length]
    for i in l:
        current_data = paddle.load(os.path.join(path,i))
        data_buffer1.append(current_data[0])
        data_buffer2.append(current_data[1])
        data_buffer3.append(current_data[2])
        count += 1 # 计数器++
        if count == batch_size:
            yield paddle.concat(data_buffer1, 0),paddle.concat(data_buffer2, 0),paddle.concat(data_buffer3, 0)
            count = 0 # 计数器清零
            del data_buffer1,data_buffer2,data_buffer3,
            data_buffer1,data_buffer2,data_buffer3 = [],[],[] # 重置缓冲区
    else: # 有可能最后几个不足batch size了，直接返回
        if not drop_last:
            yield paddle.concat(data_buffer1, 0),paddle.concat(data_buffer2, 0),paddle.concat(data_buffer3, 0)

def process_data(epoch, path, train_loader): # length = len(train_loader)
    global step
    length = len(train_loader)
    paddle.save(length,os.path.join(path,'length.pdtensor'))
    length = paddle.load(os.path.join(path,'length.pdtensor'))

    #paddle.save(length,os.path.join(path,f'length.pdtensor'))

    sum_loss, sum_samples = float('nan'), 1
    progress_bar = tqdm(train_loader)
    logger.info('processing data')
    json_logger.info('processing data')
    count = 0
    for mix_stft_mag, vocal_stft_mag, instru_stft_mag in progress_bar:
        paddle.save((mix_stft_mag, vocal_stft_mag, instru_stft_mag),os.path.join(path,f'data_{(count:=count+1)}.pdtensor'))
    print('process data finished.')

def fast_train(epoch, model_list, multi_loss, criterion, optimizer, path, params, writer):
    global step
    rank = paddle.distributed.get_rank()
    for model in model_list:
        model.train()

    # 产生数据集长度
    length = paddle.load(os.path.join(path,'length.pdtensor'))#len(train_loader)
    # 卡的数量
    world_size = paddle.distributed.get_world_size()
    # 每一张卡的数据集长度
    length = length // world_size
    # 根据batch size进行划分
    length = int(math.ceil(length/hps['batch_size']))
    
    sum_loss, sum_samples = 0, 0
    progress_bar = tqdm(get_data(path))
    batch_idx = -1
    for mix_stft_mag, vocal_stft_mag, instru_stft_mag in progress_bar:
        start_time = time.time()
        batch_idx += 1
        sum_samples += len(mix_stft_mag)

        mix_stft_mag = mix_stft_mag.transpose([0,1,3,2])
        
        separate_stft_mag = []
        vocal_stft_mag = vocal_stft_mag.transpose([0,1,3,2])
        instru_stft_mag = instru_stft_mag.transpose([0,1,3,2])
        separate_stft_mag.append(vocal_stft_mag)
        separate_stft_mag.append(instru_stft_mag)
        with paddle.amp.auto_cast(enable=params.amp, level='O1'):
            loss = multi_loss(mix_stft_mag, separate_stft_mag)

        scaled = scaler.scale(loss) # loss 缩放，乘以系数 loss_scaling
        scaled.backward()
        scaler.step(optimizer)      # 更新参数（参数梯度先除系数 loss_scaling 再更新参数）
        scaler.update()             # 基于动态 loss_scaling 策略更新 loss_scaling 系数
        optimizer.clear_grad()
        step += 1

        sum_loss += loss.item() * len(mix_stft_mag)
        progress_bar.set_description(
            'training epoch：{:3d} [{:4d}/{:4d} ({:3.3f}%)] loss：{:.4f}'.format(
                epoch,
                batch_idx + 1, length,
                100. * (batch_idx + 1) / length,
                sum_loss / sum_samples)
            )
        if not rank:
            writer.add_scalar(tag="损失", step=step, value=sum_loss / sum_samples)
        json_logger.log(step,{
                                 "train.loss" : sum_loss/sum_samples,
                                 "train.ips" : 1/(time.time() - start_time),
                                 "data.shape" : (mix_stft_mag.shape,vocal_stft_mag.shape,instru_stft_mag.shape),
                                 "train.lr" : optimizer.get_lr(),
                                 "rank" : paddle.distributed.get_rank(),
                                 "epoch" : epoch,
                             }
                        )

    if params.clean_logs:
        os.system('clear' if 'linux' in sys.platform else 'cls')

    for i in range(len(params['num_instruments'])):
        paddle.save({'epoch': epoch, 
                    'state_dict': model_list[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trainer' : params.trainer,
                    'step' : step
                    },
                   '{}/net_{}_{}.pdparams'.format(params['model_dir'], params['num_instruments'][i], epoch))
        moXingLuJing = '{}/net_{}_{}.pdparams'.format(params['model_dir'], params['num_instruments'][i], epoch)
        logger.info(f'save model to {moXingLuJing}')
        json_logger.info(data=f'save model to {moXingLuJing}')
        if len(os.listdir(params.model_dir)) > params.keep_ckpt * 2:
            epoch_list = []
            for j in os.listdir(params.model_dir):
                if len(j.split('_')) == 3 and j.split('_')[0] == 'net' and j.split('_')[1] == params['num_instruments'][i] and j.split('_')[-1].endswith('.pdparams'):
                    try:
                        model_epoch = int(j.split('_')[-1].replace('.pdparams',''))
                        epoch_list.append(model_epoch)

                    except Exception as e:
                        continue

            # 不动预训练模型
            if 0 in epoch_list:
                epoch_list.remove(0)

            # 产生要删除的回合的模型
            epoch_list.sort()
            shanChuLieBiao = epoch_list[:-params.keep_ckpt]
            for shanChuDeHuiHe in shanChuLieBiao:
                yueQi = params.num_instruments[i]
                try:
                    os.remove(os.path.join(params.model_dir,f'net_{yueQi}_{shanChuDeHuiHe}.pdparams'))
                    moXingMingCheng = f'net_{yueQi}_{shanChuDeHuiHe}.pdparams'
                    logger.info(f'deleting {moXingMingCheng} to save disk memory')
                    json_logger.info(f'deleting {moXingMingCheng} to save disk memory')
                except Exception as e:
                    luJing = os.path.join(params.model_dir,f'net_{yueQi}_{shanChuDeHuiHe}.pdparams')
                    logger.warn(f'delete {luJing} failed：' + str(e))
                    json_logger.info(f'delete {luJing} failed：' + str(e))
    r''' # 这个太费时了，不要了
    ceShiMoXing = Separator(resume = deDaoZuiXinMoXing(params))
    ceShiMoXing.eval()
    
    for meiGeCeShi in ['测试数据1','测试数据2']:
        if os.path.isfile(meiGeCeShi) and not rank:
            yinPin, caiYangLv = sf.read(meiGeCeShi)
            yinPin = paddle.to_tensor(yinPin,dtype="float32").T
            writer.add_audio(tag = f'原始数据/{meiGeCeShi}', audio_array = yinPin.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
            shuChu1, shuChu2, _ = ceShiMoXing(paddle.to_tensor(yinPin))
            writer.add_audio(tag = f'{meiGeCeShi}/轨道1', audio_array = shuChu1.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
            writer.add_audio(tag = f'{meiGeCeShi}/轨道2', audio_array = shuChu2.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
    del ceShiMoXing
    '''
    del mix_stft_mag, vocal_stft_mag, instru_stft_mag

def train(epoch, model_list, multi_loss, criterion, optimizer, train_loader, params, writer):
    global step
    rank = paddle.distributed.get_rank()
    for model in model_list:
        model.train()

    sum_loss, sum_samples = 0, 0
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (mix_stft_mag, vocal_stft_mag, instru_stft_mag) in enumerate(train_loader):
        start_time = time.time()
        sum_samples += len(mix_stft_mag)

        mix_stft_mag = mix_stft_mag.transpose([0,1,3,2])
        
        separate_stft_mag = []
        vocal_stft_mag = vocal_stft_mag.transpose([0,1,3,2])
        instru_stft_mag = instru_stft_mag.transpose([0,1,3,2])
        separate_stft_mag.append(vocal_stft_mag)
        separate_stft_mag.append(instru_stft_mag)
        with paddle.amp.auto_cast(enable=params.amp, level='O1'):
            loss = multi_loss(mix_stft_mag, separate_stft_mag)

        scaled = scaler.scale(loss) # loss 缩放，乘以系数 loss_scaling
        scaled.backward()
        scaler.step(optimizer)      # 更新参数（参数梯度先除系数 loss_scaling 再更新参数）
        scaler.update()             # 基于动态 loss_scaling 策略更新 loss_scaling 系数
        optimizer.clear_grad()
        step += 1

        sum_loss += loss.item() * len(mix_stft_mag)
        progress_bar.set_description(
            'training epoch：{:3d} [{:4d}/{:4d} ({:3.3f}%)] loss：{:.4f}'.format(
                epoch,
                batch_idx + 1, len(train_loader),
                100. * (batch_idx + 1) / len(train_loader),
                sum_loss / sum_samples)
            )
        if not rank:
            writer.add_scalar(tag="损失", step=step, value=sum_loss / sum_samples)
        json_logger.log(step,{
                                 "train.loss" : sum_loss/sum_samples,
                                 "train.ips" : 1/(time.time() - start_time),
                                 "data.shape" : (mix_stft_mag.shape,vocal_stft_mag.shape,instru_stft_mag.shape),
                                 "train.lr" : optimizer.get_lr(),
                                 "rank" : paddle.distributed.get_rank(),
                                 "epoch" : epoch,
                             }
                        )

    if params.clean_logs:
        os.system('clear' if 'linux' in sys.platform else 'cls')

    for i in range(len(params['num_instruments'])):
        # 保存当前模型
        paddle.save({'epoch': epoch, 
                    'state_dict': model_list[i].state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'trainer' : params.trainer,
                    'step' : step
                    },
                   '{}/net_{}_{}.pdparams'.format(params['model_dir'], params['num_instruments'][i], epoch))
        moXingLuJing = '{}/net_{}_{}.pdparams'.format(params['model_dir'], params['num_instruments'][i], epoch)
        logger.info(f'save model to {moXingLuJing}')
        json_logger.info(f'save model to {moXingLuJing}')
        
        if len(os.listdir(params.model_dir)) > params.keep_ckpt * 2:
            epoch_list = []

            for j in os.listdir(params.model_dir):
                if len(j.split('_')) == 3 and j.split('_')[0] == 'net' and j.split('_')[1] == params['num_instruments'][i] and j.split('_')[-1].endswith('.pdparams'):
                    try:
                        model_epoch = int(j.split('_')[-1].replace('.pdparams',''))
                        epoch_list.append(model_epoch)

                    except Exception as e:
                        continue

            # 不动预训练模型
            if 0 in epoch_list:
                epoch_list.remove(0)

            epoch_list.sort()
            shanChuLieBiao = epoch_list[:-params.keep_ckpt]
            for shanChuDeHuiHe in shanChuLieBiao:
                yueQi = params.num_instruments[i]
                try:
                    os.remove(os.path.join(params.model_dir,f'net_{yueQi}_{shanChuDeHuiHe}.pdparams'))
                    moXingMingCheng = f'net_{yueQi}_{shanChuDeHuiHe}.pdparams'
                    logger.info(f'deleting model {moXingMingCheng} to save disk memory...')
                    json_logger.info(f'deleting model {moXingMingCheng} to save disk memory...')
                except Exception as e:
                    luJing = os.path.join(params.model_dir,f'net_{yueQi}_{shanChuDeHuiHe}.pdparams')
                    logger.warn(f'delete old model {luJing} failed：' + str(e))
                    json_logger.info(f'delete old model {luJing} failed：' + str(e))

    ceShiMoXing = Separator(resume = deDaoZuiXinMoXing(params))
    ceShiMoXing.eval()
        
    for meiGeCeShi in ['测试数据1','测试数据2']:
        if os.path.isfile(meiGeCeShi) and not rank:
            yinPin, caiYangLv = sf.read(meiGeCeShi)
            yinPin = paddle.to_tensor(yinPin,dtype="float32").T
            writer.add_audio(tag = f'原始数据/{meiGeCeShi}', audio_array = yinPin.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
            shuChu1, shuChu2, _ = ceShiMoXing(yinPin)
            writer.add_audio(tag = f'{meiGeCeShi}/轨道1', audio_array = shuChu1.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
            writer.add_audio(tag = f'{meiGeCeShi}/轨道2', audio_array = shuChu2.mean(0).detach().numpy(), step = step, sample_rate = caiYangLv)
    
    del ceShiMoXing,mix_stft_mag, vocal_stft_mag, instru_stft_mag

def main(params):
    global step
    paddle.seed(params.seed)
    train_dataset = TrainDataset(params)
    n_chunks = train_dataset.count
    logger.info('chunk size：{}'.format(n_chunks))
    json_logger.info('chunk size：{}'.format(n_chunks))

    model_list = nn.LayerList()
    start = 1

    if (params.load_model == True) and (None not in deDaoZuiXinMoXing(params)):
        logger.info('=>load model {}'.format(deDaoZuiXinMoXing(params)))
        json_logger.info('=>load model {}'.format(deDaoZuiXinMoXing(params)))
        try:
            for i in range(len(params.num_instruments)):
                checkpoint = paddle.load(deDaoZuiXinMoXing(params)[i])
                if i == 0:
                    logger.info('loaded model success, trainer：' + checkpoint['trainer'] + '，epoch：' + str(checkpoint['epoch']) + '，step：' + str(checkpoint['step']))
                    json_logger.info('loaded model success, trainer：' + checkpoint['trainer'] + '，epoch：' + str(checkpoint['epoch']) + '，step：' + str(checkpoint['step']))
                net = UNet()
                start = checkpoint['epoch'] + 1
                step = checkpoint['step']

                net.set_state_dict(checkpoint['state_dict'])
                net.to(device)
                net = DDP(net)
                model_list.append(net)

        except:
            logger.info(f'use new model, trainer：{params.trainer}')
            json_logger.info(f'use new model, trainer：{params.trainer}')
            for i in range(len(params['num_instruments'])):
                net = UNet()
                net.to(device)
                net = DDP(net)
                model_list.append(net)
    else:
        logger.info(f'use new model, trainer：{params.trainer}')
        json_logger.info(f'use new model, trainer：{params.trainer}')
        for i in range(len(params['num_instruments'])):
            net = UNet()
            net.to(device)
            net = DDP(net)
            model_list.append(net)

    if params['loss'] == 'l1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    multi_loss = MultiLoss(model_list, criterion, params)
    if params['optimizer'] == 'sgd':
        optimizer = paddle.optimizer.SGD(parameters = multi_loss.parameters(), learning_rate = params['lr'], weight_decay=params['wd']) # , momentum=params['momentum'], dampening=params['dampening']
    elif params['optimizer'] == 'adagrad':
        optimizer = paddle.optimizer.Adagrad(parameters = multi_loss.parameters(), learning_rate = params['lr'], weight_decay=params['wd']) # , lr_decay=params['momentum']
    else:
        optimizer = paddle.optimizer.Adam(parameters = multi_loss.parameters(), learning_rate = params['lr'], weight_decay = params['wd'])
    if params['load_optimizer'] and not (checkpoint['optimizer'] is None):
        try:
            optimizer.set_state_dict(checkpoint['optimizer'])
        except:
            pass
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                              num_workers=psutil.cpu_count())
    writer = LogWriter(logdir="./logs")

    for epoch in range(start, params['epochs']):
        # train(epoch, model_list, multi_loss, criterion, optimizer, train_loader, params, writer)
        # process_data(-1,'train_data',train_loader)
        # fast_train(epoch, model_list, multi_loss, criterion, optimizer, 'train_data', params, writer)

        if len(sys.argv) == 1:
            raise ValueError("please input correct arguments！！！")

        if sys.argv[1] == "train":
            train(epoch, model_list, multi_loss, criterion, optimizer, train_loader, params, writer)
        elif sys.argv[1] == "process_data" and len(sys.argv) == 2 + 1:
            process_data(-1,'train_data',train_loader)
            break
        elif sys.argv[1] == "fast_train":
            fast_train(epoch, model_list, multi_loss, criterion, optimizer, 'train_data', params, writer)
        elif "fast_train" in sys.argv and "process_data" in sys.argv:
            if epoch == start:
                process_data(-1,'train_data',train_loader)
            fast_train(epoch, model_list, multi_loss, criterion, optimizer, 'train_data', params, writer)
        else:
            raise ValueError("please input correct arguments！！！")


if __name__ == '__main__':
    try:
        hps = HParams(**eval(sys.argv[-1]))
        os.makedirs(hps['model_dir'], exist_ok=True)
        main(hps)
    except Exception as e:
        raise e
