import torch
import torch.nn as nn
import numpy as np
from kmeans_pytorch import kmeans


class RoutingAttention(nn.Module):
    def __init__(self, heads):
        super().__init__()

        self.n_heads = heads
    
    def forward(self, qk):
        qk = qk.detach()
        batch_size, seq_len, dim, device = *qk.shape, qk.device # [6,12,384,64]

        W_R = torch.empty(batch_size, dim, dim, device=device)
        nn.init.orthogonal_(W_R)
        
        R = torch.matmul(qk, W_R).reshape(-1, dim)
        K = int(seq_len ** 0.5)
        
        cluster_idx, centroid = kmeans(X=R, num_clusters=K, distance='euclidean', device=device)
        
        cluster_idx = cluster_idx.reshape(batch_size, seq_len).unsqueeze(1).expand(-1, self.n_heads, -1)
        result = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=device)
        r1 = result + cluster_idx.unsqueeze(-1)   # [0, 0, 0, 0, ...]
        r2 = result + cluster_idx.unsqueeze(-2)   # [0, 1, 2, 3, ...]

        result = (r1 == r2).to(torch.float32)
        result = 10000. * result - 10000. 
        
        return result.detach()
        
