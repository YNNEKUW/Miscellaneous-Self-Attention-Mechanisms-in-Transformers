import torch
import torch.nn as nn


class SimpleLSHAttention16(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk, bucket_size=32, **kwargs):
        qk = qk.detach()
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device 

        qk_norm = qk.div(torch.norm(qk, dim=-1, keepdim=True))
        qk_const = torch.norm(qk_norm, dim=-1, keepdim=True)

        qk_const = torch.sqrt(1. - torch.pow(qk_const, 2))
        qk = torch.cat((qk, qk_const), -1)
        a = torch.randn([batch_size, n_heads, seq_len, dim+1], device=device).normal_(mean=0, std=1)
        Q = torch.sum(qk.mul(a), dim=-1)           
        P = torch.matmul(qk, a.transpose(-1, -2)).permute(3, 0, 1, 2)

        
        result = Q.unsqueeze(0).mul(P)                             
        result = result.permute(1, 2, 0, 3)
        # Deal with Nan
        result[result!=result] = 0.
        max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0.)

        return result.detach()


