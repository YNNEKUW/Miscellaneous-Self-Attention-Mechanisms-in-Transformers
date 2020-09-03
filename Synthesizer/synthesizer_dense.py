import torch
from torch import nn


class Dense_projection(nn.Module):
    def __init__(self, max_seq_len, attention_head_size, bottleneck_size):
        super().__init__()
        self.dense1 = nn.Linear(attention_head_size, bottleneck_size)
        self.dense2 = nn.Linear(bottleneck_size, max_seq_len)
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, hidden_states, input_len):
        res = self.act(self.dense1(hidden_states))
        res = self.dense2(res)[:, :, :, :input_len]
        return res