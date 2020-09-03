import torch
from torch import nn


class Rand_weight(nn.Module):
    def __init__(self, max_seq_len, num_attention_heads):
        super(Rand_weight, self).__init__()

        rand_att = list()
        rand_att.append(torch.eye(max_seq_len))
        rand_att.append(torch.roll(torch.eye(max_seq_len), 1, -1))
        rand_att.append(torch.roll(torch.eye(max_seq_len), -1, -1))

        tmp = torch.pow(torch.range(0, max_seq_len-1)+1, 3).unsqueeze(0).expand(max_seq_len, -1)
        mask = torch.tril(torch.roll(torch.tril(1.0-torch.eye(max_seq_len)), -1, -1))
        mask[-1][-1] = 0.0
        tm = tmp*mask
        
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)
        
        mask = torch.triu(torch.roll(torch.triu(1.0-torch.eye(max_seq_len)), 1, -1))
        mask[1][1] = 0.0
        tm = tmp*mask
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        tm = tmp
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        tmp = torch.pow(torch.flip(torch.range(0, max_seq_len-1)+1, (0,)), 3).unsqueeze(0).expand(max_seq_len, -1)

        tm = tmp
        sum_ = tm.sum(axis=-1)
        sum_[torch.where(sum_ == 0)] = 1
        tm = tm / sum_.unsqueeze(-1)
        rand_att.append(tm)

        R = torch.stack(rand_att)

        if num_attention_heads > 7:
            self.R = torch.cat((R, torch.randn((num_attention_heads-7, max_seq_len, max_seq_len)).normal_(mean=0, std=1)), dim=0)
        else:
            self.R = R
        self.R = nn.Parameter(self.R, requires_grad=True)

    def forward(self, input_len):
        return self.R[:, :input_len, :input_len].unsqueeze(0)
