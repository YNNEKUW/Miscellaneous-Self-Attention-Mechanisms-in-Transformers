import torch
import torch.nn as nn


class XBOXAttention16(nn.Module):
    def __init__(self, qnf):
        super().__init__()
        self.QNF = qnf

    def forward(self, qk, bucket_size=32, **kwargs):
        qk = qk.detach().to(torch.float16)
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device

        qk_norm = torch.norm(qk, dim=-1, keepdim=True)
        phi = torch.max(qk_norm)

        qk_const = torch.sqrt(torch.pow(phi, 2) - torch.pow(qk_norm, 2))
        tmp_zero = torch.zeros(qk_const.shape, device=device, dtype=torch.float16)

        Q = torch.cat((qk, tmp_zero), -1)
        P = torch.cat((qk, qk_const), -1)
        
        if self.QNF:
            _P_norm = torch.norm(P, dim=-1, keepdim=True)
            _Q_norm = torch.norm(Q, dim=-1, keepdim=True)
            _M = torch.max(_P_norm)
            P = (P / _P_norm) * _M
            Q = (Q / _Q_norm) * _M
        
        if P.shape != Q.shape:
            raise ValueError("Shapes of P and Q are mismatch.")

        a = torch.randn([batch_size, n_heads, seq_len, dim+1], device=device).to(torch.float16).normal_(mean=0, std=1)
        
        Q = torch.sum(Q.mul(a), dim=-1)           
        P = torch.matmul(P, a.transpose(-1, -2)).permute(3, 0, 1, 2)


        result = Q.unsqueeze(0).mul(P)                             
        result = result.permute(1, 2, 0, 3)
        
        result[result!=result] = 0.
        max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0)
        return result.detach()