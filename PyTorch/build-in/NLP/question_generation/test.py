import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("sdaa")

batch_size, seq_len = 512, 512
embedding_dim = 8
num_weights = 32
padding_idx = -1
scale_grad_by_freq = False

embedding = nn.Embedding(
    num_embeddings=num_weights,
    embedding_dim=embedding_dim,
    padding_idx=padding_idx if padding_idx >= 0 else None
).to(device)

indices = torch.randint(0, num_weights, (batch_size, seq_len), device=device)

output = embedding(indices)
print("="*100)
output.backward(torch.ones_like(output).as_strided(output.shape,[512, 1, 262144]))