#encoding=utf-8
import os
import yaml
import argparse

parser = argparse.ArgumentParser(description='launch train.py with some parameters')
parser.add_argument('--lr', help='学习率', type=float,
                    required=False,default=0.0005)

parser.add_argument('--bs', help='批次大小', type=int,
                    required=False,default=64)

parser.add_argument('-il','--interval_log', help='日志记录间隔', type=int,
                    required=False,default=100)

parser.add_argument('-iv','--interval_val', help='保存模型间隔', type=int,
                    required=False,default=200)

parser.add_argument('-ifs','--interval_force_save', help='保存不会被删除的模型间隔', type=int,
                    required=False,default=1000)

parser.add_argument('--epoch', help='训练回合数', type=int,
                    required=False,default=100000)

parser.add_argument('--num_workers', help='加载数据数量', type=int,
                    required=False,default=0)

parser.add_argument('--amp_dtype', help='混合精度训练数据类型', type=str,
                    required=False,default="fp32")

parser.add_argument('--nproc_per_node', help='DDP卡的数量', type=int,
                    required=False,default=1)

parser.add_argument('--model_name', help='使用的模型名称，不可修改', type=str,
                    required=False,default="CFNaiveMelPE")

args = parser.parse_args()

with open("config.yaml", 'r', encoding="utf-8") as file:
    data = yaml.safe_load(file)

data["train"]["lr"] = args.lr
data["train"]["batch_size"] = args.bs
data["train"]["interval_log"] = args.interval_log
data["train"]["interval_val"] = args.interval_val
data["train"]["interval_force_save"] = args.interval_force_save
data["train"]["epochs"] = args.epoch
data["train"]["num_workers"] = args.num_workers
data["train"]["amp_dtype"] = args.amp_dtype
data["model"]["type"] = args.model_name

with open('config_user.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)

devices = ",".join([str(i) for i in range(args.nproc_per_node)])
#os.system("python train.py")
if __name__ == "__main__":
    if args.nproc_per_node == 1:
        os.system(f"python train.py -c config_user.yaml")
    elif args.nproc_per_node > 1:
        os.system(f"python -m paddle.distributed.launch --devices={devices} train.py -c config_user.yaml")
    elif args.nproc_per_node < 1:
        raise ValueError("nproc_per_node cannot be less than 1!")