import torch
import torch.nn as nn
import math


class IALSHAttention16(nn.Module):
    def __init__(self,
                 qnf,
                 m=2,
                 U=0.75
                 ):
        super().__init__()
        self.m = m
        self.U = U
        self.QNF = qnf

    def forward(self, qk, bucket_size=32, **kwargs):
        qk = qk.detach().to(torch.float16)
        batch_size, n_heads, seq_len, dim, device = *qk.shape, qk.device
        # Removing Dependency on Norm of Query
        qk_norm = torch.norm(qk, dim=-1, keepdim=True)
        
        M = torch.max(qk_norm)
        qk = qk * self.U / M
        tmp_zero = torch.zeros(qk_norm.shape, device=device, dtype=torch.float16)
        
        P = qk
        Q = qk

        for i in range(self.m):
            tmp = 0.5 - torch.pow(qk_norm, (i+1)*2)
            P = torch.cat((P, tmp), -1)
            
            Q = torch.cat((Q, tmp_zero), -1)
        
        if self.QNF:
            _P_norm = torch.norm(P, dim=-1, keepdim=True)
            _Q_norm = torch.norm(Q, dim=-1, keepdim=True)
            _M = torch.max(_P_norm)
            P = (P / _P_norm) * _M
            Q = (Q / _Q_norm) * _M
        
        if P.shape != Q.shape:
            raise ValueError('Shapes of P and Q are mismatch.')
        
        a = torch.randn([batch_size, n_heads, seq_len, dim+self.m], device=device).to(torch.float16).normal_(mean=0, std=1)

        Q = torch.sum(Q.mul(a), dim=-1)           
        P = torch.matmul(P, a.transpose(-1, -2)).permute(3, 0, 1, 2)


        result = Q.unsqueeze(0).mul(P)                           
        result = result.permute(1, 2, 0, 3)

        result[result!=result] =0.
        max_idx = torch.topk(result, k=bucket_size, dim=-1)[1]
        result = torch.ones(result.shape, device=device) * (-10000.)
        result.scatter_(-1, max_idx, 0.)
        return result.detach()
    
      
class IALSHSelfAttention(nn.Module):
    def __init__(self,
                 dim,      # config.hidden_size
                 heads=12, # default LSH is 8, config.num_attention_heads
                 output_attentions=False,    # config.output_attentions
                 attention_probs_dropout_prob=0.1):   #config.attention_probs_dropout_prob
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"The hidden size {dim} is not a multiple of the number of attention.")

        self.dim = dim
        self.num_attention_heads = heads
        self.attention_head_size = int(self.dim / self.num_attention_heads)
        self.output_attentions = output_attentions
        
        # No implement v_heads here
        # self.v_head_repeats = (heads if one_value_head else 1)
        

        self.attention_head_size = int(dim / heads)
        # self.all_head_size = self.num_attention_heads * self.attention_head_size

        # multi-head attention here, remove the original "one_value_head"
        
        self.qk = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.ialsh_attn = IALSHAttention(m=2,U=0.75)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # (6,384,12,64)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):
        # (6,384,768)
        device, dtype = hidden_states.device, hidden_states.dtype
        b, l, d, h = *hidden_states.shape, self.num_attention_heads

        mixed_qk_layer = self.qk(hidden_states)

        # Not dealing with "encoder_hidden_states" here
        """
        if encoder_hidden_sates is not None:
            mixed_value_layer = self.value(enocer_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_value_layer = self.value(hidden_states)
        """
        mixed_value_layer = self.value(hidden_states)
        """
        def _merge_heads(v):
            return v.view(b, l, h, -1).transpose(1, 2).reshape(b * h, l, -1)
        
        def _split_heads(v):
            return v.view(b, h , l, -1).transpose(1, 2).contiguous()
        
        mixed_qk_layer = _merge_heads(mixed_qk_layer) # (6, 384, 768)->(6*12, 384, 64)
        mixed_value_layer = _merge_heads(mixed_value_layer)
        """
        qk_layer = self.transpose_for_scores(mixed_qk_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #
        # modify the attention here
        #
        # mixed_value_layer = mixed_value_layer.repeat(1, 1, self.v_head_repeats)
        #
        # dropout before qk*v
        # masks = {}
        attention_scores = torch.matmul(qk_layer, qk_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # self.ialsh_attn(qk_layer)
        attention_hash_mask = self.ialsh_attn(qk_layer)
        attention_scores = attention_scores + attention_hash_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [6, 384, 12, 64]
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        
        return outputs

