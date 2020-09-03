import torch
import torch.nn as nn
import numpy as np
import math
from scipy.linalg import block_diag
import pdb


class SparseAttentionStrided(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk):
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device
        l = math.floor(seq_len ** 0.5)
        local_mask = torch.ones(seq_len, seq_len, device=device)
        local_mask = torch.triu(local_mask, -l).permute(1, 0)
        local_mask = torch.triu(local_mask, -l)
        local_mask = torch.where(local_mask==1, torch.tensor(0., device=device), torch.tensor(-10000., device=device))
        local_mask = torch.unsqueeze(torch.unsqueeze(local_mask, 0), 0).expand(batch_size, int(n_heads/2), -1, -1)

        
        x = torch.arange(seq_len, device=device).unsqueeze(-1)
        y = x.permute(1, 0)
        z = torch.zeros(seq_len, seq_len, device=device)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = (torch.fmod(q-k, l) == 0)
        # global_mask = (c1 * c2).to(torch.float32)
        global_mask = c2.to(torch.float32)
        global_mask = torch.where(global_mask==1, torch.tensor(0., device=device), torch.tensor(-10000., device=device))
        global_mask = torch.unsqueeze(torch.unsqueeze(global_mask, 0), 0).expand(batch_size, int(n_heads/2), -1, -1)

        result = torch.cat((local_mask, global_mask), 1)

        return result.detach()


        
class SparseAttentionFixed(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, qk):
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device
        l = math.floor(seq_len ** 0.5)
        num_of_block = math.ceil(seq_len / l)

        small_block = np.ones([l, l])
        _tmp = [small_block for _ in range(num_of_block)]
        local_mask = block_diag(*_tmp)
        
        local_mask = local_mask[:seq_len, :seq_len] * 10000. - 10000.
        local_mask = torch.tensor(local_mask, device=device, dtype=torch.float32)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, int(n_heads/2), -1, -1)

        global_mask = torch.zeros(seq_len, seq_len, device=device)
        global_mask[:, l-1::l] = 1.
        global_mask = global_mask * 10000. -10000.
        global_mask = global_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, int(n_heads/2), -1, -1)
        
        result = torch.cat((local_mask, global_mask), 1)

        return result.detach()


        
        


