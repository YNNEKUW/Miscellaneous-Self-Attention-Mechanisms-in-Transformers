import torch
from torch import nn
from LSH.IALSH import IALSHAttention16
from LSH.SimpleLSH import SimpleLSHAttention16
from LSH.SimpleALSH import SimpleALSHAttention16
from LSH.XBOX import XBOXAttention16
from Sparse.SparseAttention import SparseAttentionStrided, SparseAttentionFixed
from Synthesizer.synthesizer_dense import Dense_projection
from Synthesizer.synthesizer_random import Rand_weight
from RoutingTransformer.RoutingAttention import RoutingAttention
import math


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, max_seq_len, attention_name, bucket_size=None, output_attentions=False, keep_multihead_output=False, attention_probs_dropout_prob=0.1):
        super(SelfAttention, self).__init__()

        self.output_attentions = output_attentions
        self.keep_multihead_output = keep_multihead_output

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.value = nn.Linear(hidden_size, self.all_head_size)

        attentions = {'dense': Dense_projection(max_seq_len, self.attention_head_size), 'rand': Rand_weight(max_seq_len, self.num_attention_heads), 'routing': RoutingAttention(self.num_attention_heads), 'ia': IALSHAttention16(False), 'simple': SimpleLSHAttention16(), 'simple_a': SimpleALSHAttention16(), 'ia_QNF': IALSHAttention16(True), 'xbox': XBOXAttention16(False), 'xbox_qnf': XBOXAttention16(True), 'sparse': SparseAttentionStrided(), 'sparse_f': SparseAttentionFixed()}

        self.bucket_size = bucket_size
        self.attention_name = attention_name
        self.attention = attentions[self.attention_name]

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x, num_attention_heads=None, attention_head_size=None):
        if num_attention_heads:
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):

        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if self.bucket_size:
            mixed_query_layer = self.query(hidden_states)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            if query_layer.shape[-2] > self.bucket_size:
                hash_mask = self.attention(query_layer, attention_mask, self.bucket_size)
            else:
                hash_mask = torch.zeros(attention_mask.shape).to(query_layer.device)
            attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
            attention_scores += hash_mask 
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        else:
            input_seq_len = hidden_states.shape[1]
            if self.attention_name == 'rand': 
                attention_scores = self.attention(input_seq_len)
            elif self.attention_name == 'dense':
                attention_scores = self.attention(self.transpose_for_scores(hidden_states), input_seq_len)
            else:
                mixed_query_layer = self.query(hidden_states)
                query_layer = self.transpose_for_scores(mixed_query_layer)
                hash_mask = self.attention(query_layer)
                attention_scores = torch.matmul(query_layer, query_layer.transpose(-1, -2))
                attention_scores += hash_mask
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        if self.keep_multihead_output: 
            self.multihead_output = context_layer
            self.multihead_output.retain_grad()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if self.output_attentions:
            return attention_probs, context_layer
        return context_layer