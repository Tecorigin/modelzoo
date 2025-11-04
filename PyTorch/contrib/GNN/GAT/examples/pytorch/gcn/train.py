import argparse
import dgl
import dgl.nn as dglnn
import torch
import torch_sdaa
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os
# 设置IP:PORT，框架启动TCP Store为ProcessGroup服务
os.environ['MASTER_ADDR'] = '127.0.0.1' # 设置IP
# 从外部获取local_rank参数
local_rank = int(os.environ.get("LOCAL_RANK", -1))
class DenseGCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hid_size)
        self.linear2 = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, adj, features):
        # First GCN layer
        h = self.linear1(features)
        h = torch.mm(adj, h)
        h = F.relu(h)
        
        # Dropout
        h = self.dropout(h)
        
        # Second GCN layer
        h = self.linear2(h)
        h = torch.mm(adj, h)
        return h

def evaluate(adj, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(adj, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def train(args,adj, features, labels, masks, model,rank):
    if rank == 0:
        log_file_path = './scripts/train_model_log_'+args.dataset+'.txt'
        model_train_log_file = open(log_file_path, 'w')
    if args.use_amp:
        scaler = torch.amp.GradScaler('sdaa') 
    
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    if args.use_amp:
        for epoch in range(200):
            model.train()
            with torch.sdaa.amp.autocast():  
                logits = model(adj, features)
                loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            scaler.scale(loss).backward()    # loss缩放并反向转播
            scaler.step(optimizer)    # 参数更新
            scaler.update()
            acc = evaluate(adj, features, labels, val_mask, model)
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
            if rank == 0:
                model_train_log_file.write(f'Epoch: {epoch} Loss: {loss.item()}  acc: {acc}\n')
                model_train_log_file.flush()
    else:
        
        for epoch in range(200):
            model.train()
            logits = model(adj, features)
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = evaluate(adj, features, labels, val_mask, model)
            print(
                "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                    epoch, loss.item(), acc
                )
            )
            if rank == 0:
                model_train_log_file.write(f'Epoch: {epoch} Loss: {loss.item()}  acc: {acc}\n')
                model_train_log_file.flush()
    if rank == 0:
        model_train_log_file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--use_ddp",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with Dense GCN implementation.")

    # DDP backend初始化
    # rank号和device一一对应，进行set device
    # 需要在初始化ProcessGroup之前进行set device，tccl限制
    if args.use_ddp:
        device = torch.device(f"sdaa:{local_rank}")
        torch.sdaa.set_device(device)
        # 初始化ProcessGroup，通信后端选择tccl
        torch.distributed.init_process_group(backend="tccl", init_method="env://")
    else:
        device = torch.device("sdaa" if torch.sdaa.is_available() else "cpu")
        
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    # load and preprocess dataset
    transform = AddSelfLoop()
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    
    g = data[0]
    
    
    # Convert sparse adjacency matrix to dense
    adj_sparse = g.adjacency_matrix(transpose=True)
    adj_dense = adj_sparse.to_dense()
    
    # Normalize adjacency matrix
    deg = torch.sum(adj_dense, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    adj_norm = torch.mul(torch.mul(adj_dense, deg_inv_sqrt.unsqueeze(1)), deg_inv_sqrt.unsqueeze(0))
    
    # Move everything to SDAA device
    adj_norm = adj_norm.to(device)
    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)
    masks = g.ndata["train_mask"].to(device), g.ndata["val_mask"].to(device), g.ndata["test_mask"].to(device)

    # create GCN model
    in_size = features.shape[1]

    out_size = data.num_classes
    model = DenseGCN(in_size, 16, out_size).to(device)

    # convert to bfloat16 if needed
    '''if args.dt == "bfloat16":
        adj_norm = adj_norm.to(dtype=torch.bfloat16)
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)'''
    if args.use_ddp:
        model = DDP(model)
    else:
        if args.dt == "bfloat16":
            adj_norm = adj_norm.to(dtype=torch.bfloat16)
            features = features.to(dtype=torch.bfloat16)
            model = model.to(dtype=torch.bfloat16)
    # model training
    print("Training...")
    train(args,adj_norm, features, labels, masks, model,rank)

    # test the model
    print("Testing...")
    acc = evaluate(adj_norm, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))
