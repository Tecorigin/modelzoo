import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

class DenseGATLayer(nn.Module):
    def __init__(self, in_size, out_size, heads=1, feat_drop=0.6, attn_drop=0.6, activation=None):
        super().__init__()
        self.heads = heads
        self.out_size = out_size
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = activation
        
        # Linear transformations for each head
        self.fc = nn.Linear(in_size, heads * out_size, bias=False)
        # Attention weights for each head
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, heads, 2 * out_size)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
    
    def forward(self, adj, h):
        # Apply feature dropout
        h = self.feat_drop(h)
        
        # Linear transformation
        feat = self.fc(h).view(-1, self.heads, self.out_size)  # [N, H, F]
        
        # Calculate attention scores
        # Prepare the concatenated features
        f_src = feat
        f_dst = feat
        
        # Reshape for attention calculation
        el = (torch.cat([f_src, f_dst], dim=2).permute(1, 0, 2))  # [H, N, 2F]
        er = (torch.cat([f_dst, f_src], dim=2).permute(1, 0, 2))  # [H, N, 2F]
        
        # Compute attention scores
        e = F.leaky_relu(torch.bmm(el, self.attn.permute(1, 2, 0)) + 
                        torch.bmm(er, self.attn.permute(1, 2, 0)))  # [H, N, 1]
        e = e.permute(1, 0, 2)  # [N, H, 1]
        
        # Convert sparse adjacency matrix to dense
        adj_dense = adj.to_dense()
        
        # Mask attention scores using adjacency matrix
        attention = torch.where(
            adj_dense.unsqueeze(1) > 0,
            e,
            float('-inf')
        )
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)
        attention = self.attn_drop(attention)
        
        # Apply attention to features
        h_prime = torch.bmm(attention.transpose(1, 2), 
                          feat.view(-1, self.heads, self.out_size))
        
        # Apply activation if specified
        if self.activation:
            h_prime = self.activation(h_prime)
            
        return h_prime

class DenseGAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # Two-layer GAT
        self.gat_layers.append(
            DenseGATLayer(
                in_size,
                hid_size,
                heads=heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            DenseGATLayer(
                hid_size * heads[0],
                out_size,
                heads=heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, adj, features):
        h = features
        for i, layer in enumerate(self.gat_layers):
            h = layer(adj, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
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

def train(adj, features, labels, masks, model):
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed').",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with Dense GAT implementation.")

    # Load dataset
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
        
    g = data[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get adjacency matrix
    adj = g.adjacency_matrix().to(device)
    
    # Get node features and labels
    features = g.ndata["feat"].to(device)
    labels = g.ndata["label"].to(device)
    masks = g.ndata["train_mask"].to(device), g.ndata["val_mask"].to(device), g.ndata["test_mask"].to(device)

    # Create model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = DenseGAT(in_size, 8, out_size, heads=[8, 1]).to(device)

    # Convert to bfloat16 if specified
    if args.dt == "bfloat16":
        features = features.to(dtype=torch.bfloat16)
        model = model.to(dtype=torch.bfloat16)

    # Train model
    print("Training...")
    train(adj, features, labels, masks, model)

    # Test model
    print("Testing...")
    acc = evaluate(adj, features, labels, masks[2], model)
    print("Test accuracy {:.4f}".format(acc))