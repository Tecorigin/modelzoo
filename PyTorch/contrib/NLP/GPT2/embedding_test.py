import torch
import torch_sdaa

import torch.nn as nn
import torch.optim as optim

device = "sdaa:0"

def test_embedding():
    vocab_size = 32  
    embed_dim = 8    
    batch_size = 1024  
    seq_len = 1024

    embedding = nn.Embedding(vocab_size, embed_dim,device=device)

    input_indices = torch.randint(0, vocab_size, (batch_size, seq_len),device=device)

    target = torch.randn(batch_size, seq_len, embed_dim,device=device)
    loss_fn = nn.MSELoss()
    # print("Input indices:\n", input_indices)

    embedded_output = embedding(input_indices)
    # print("Embedded output (forward pass):\n", embedded_output)

    loss = loss_fn(embedded_output, target)
    print(f"Loss: {loss.item()}")

    loss.backward()

    optimizer = optim.SGD(embedding.parameters(), lr=0.01)

    optimizer.step()
if __name__ == '__main__':
    test_embedding()
