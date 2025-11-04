# Adapted to tecorigin hardware
# 导入相关类
import argparse, os
import torch, torch_sdaa
import torch.distributed as dist
from utils import *
from DL_ClassifierModel import *

def setup_distributed(rank, world_size):
    import torch_sdaa.core.sdaa_model as sm
    sm.set_device(rank)
    dist.init_process_group(backend='tccl', rank=rank, world_size=world_size)


def cleanup_distributed():
    # 清理分布式环境
    dist.destroy_process_group()


def main():
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description='Train a protein secondary structure predictor model.')
    parser.add_argument('--train_size', type=int, default=64, help='Size of the training set.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--stop_rounds', type=int, default=100, help='Rounds to stop after no improvement.')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping rounds.')
    parser.add_argument('--save_rounds', type=int, default=1, help='Rounds to save the model.')
    parser.add_argument('--save_path', type=str, default='model/FinalModel', help='Path to save the model.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--augmentation', type=float, default=0.1, help='Data augmentation ratio.')
    parser.add_argument('--k_fold', type=int, default=3, help='Number of folds for cross-validation.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on.')
    parser.add_argument('--ddp', action='store_true', help='Use Distributed Data Parallel training.')
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node (GPUs to use)")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--data_seq", type=str, default='./data_seq_train.txt', help='Path to the dataset.')
    parser.add_argument("--data_sec", type=str, default='./data_sec_train.txt', help='Path to the dataset.')


    # 解析命令行参数
    args = parser.parse_args()

    # 如果启用了ddp，设置分布式训练环境
    if args.ddp:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup_distributed(rank, world_size)
    else:
        rank = None

    # 初始化数据类
    dataClass = DataClass(args.data_seq, args.data_sec, k=1, validSize=0.3, minCount=0)
    # 词向量预训练
    dataClass.vectorize(method='char2vec', feaSize=25, sg=1)
    # onehot+理化特征获取
    dataClass.vectorize(method='feaEmbedding')
    # 初始化模型对象
    model = FinalModel(classNum=dataClass.classNum, embedding=dataClass.vector['embedding'], feaEmbedding=dataClass.vector['feaEmbedding'], 
                    useFocalLoss=True, device=torch.device(args.device))
    
    # 如果启用了ddp，使用DDP包装模型
    if args.ddp:
        model = DistributedFinalModel(model)

    # 开始训练
    model.cv_train(dataClass, trainSize=args.train_size, batchSize=args.batch_size, epoch=args.epoch, stopRounds=args.stop_rounds, earlyStop=args.early_stop, saveRounds=args.save_rounds,
                savePath=args.save_path, lr=args.lr, augmentation=args.augmentation, kFold=args.k_fold)
    
    # 清理分布式环境
    if args.ddp:
        cleanup_distributed()

if __name__ == '__main__':
    main()