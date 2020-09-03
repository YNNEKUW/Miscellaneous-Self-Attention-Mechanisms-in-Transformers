import torch
from IALSH import IALSHSelfAttention, IALSHAttention, IALSHAttention16
from SimpleLSH import SimpleLSHAttention16, SimpleLSHAttention
from SimpleALSH import SimpleALSHAttention16
from XBOX import XBOXAttention16
from SparseAttention import SparseAttentionStrided, SparseAttentionFixed
from RoutingAttention import RoutingAttention
import pdb
import time

hidden_states = torch.randn(6, 12, 384, 64)
# hidden_states_K = torch.randn(6,384,768)

#att = IALSHAttention16().cuda()

# att = SparseAttentionFixed()
att = SimpleALSHAttention16()

#att32 = IALSHAttention().cuda()

y = att(hidden_states)
pdb.set_trace()
#y = att32(hidden_states, QNF=True)
"""
start = time.time()
y32 = att32(hidden_states)
print(time.time()-start)
print(torch.sum(y_==y32)/(6*12*384*384))
"""
"""
hidden_states = torch.randn(6, 384, 768)
attention_mask = torch.zeros(6,1,1,384)
attention_mask[0][0][0][-100:]=-10000.0
att = IALSHSelfAttention(768)
y = att(hidden_states, attention_mask=attention_mask)
"""

"""
from reformer_pytorch import LSHAttention, LSHSelfAttention, Reformer

attn = LSHAttention(bucket_size=64,n_hashes=16,causal=True)
qk = torch.randn(10,1024,128)
v = troch.randn(10,1024,128)

out, attn, buckets = attn(qk, v)
attn = LSHSelfAttention(
        dim=128,
        heads=8,
        bucket_size=64,
        n_hashes=8,
        causal=False)

x = torch.randn(10,1024,128)
y = attn(x)
model = Reformer(dim=512, depth=1, n_hashes=1, max_seq_len=384, heads=8, lsh_dropout=0.1, causal=True).cuda()
x = torch.randn(7,384,512).cuda()
y = model(x)
"""
