import torch
import torch.nn as nn


class SimpleALSHAttention16(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk, bucket_size=32, **kwargs):
        qk = qk.detach()
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device 

        M = torch.max(torch.norm(qk, dim=-1))
        qk_norm = qk / M
        qk_const = torch.norm(qk_norm, dim=-1, keepdim=True)


        qk_const = torch.sqrt(1. - torch.pow(qk_const, 2))
        tmp_zeros = torch.zeros(qk_const.shape, device=device)

        q = torch.cat((qk_norm, qk_const, tmp_zeros), -1)
        p = torch.cat((qk_norm, tmp_zeros, qk_const), -1)

        a = torch.randn([batch_size, n_heads, seq_len, dim+2], device=device).normal_(mean=0, std=1)
        Q = torch.sum(q.mul(a), dim=-1)         
        
        
        P = torch.matmul(p, a.transpose(-1, -2)).permute(3, 0, 1, 2)

        
        result = Q.unsqueeze(0).mul(P)                 
        result = result.permute(1, 2, 0, 3)
        # Deal with Nan
        result[result!=result] = 0.
        max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0.)
        
        return result.detach()




