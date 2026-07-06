import tinygrad
from tinygrad import Tensor
import math

from .mlp import Linear
from .rmsnorm import RMSNorm

def scaled_dot_product_attn(Q:Tensor, K:Tensor, V:Tensor, mask:Tensor = None):
    B, h, seq_len , d_k = Q.shape
    #print(d_k)
    attn_logits = Q.matmul(K.transpose(dim0=-2,dim1=-1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
    #print(attn_logits.numpy())
    attn_weights = attn_logits.softmax(axis=-1)
    #print(attn_weights.numpy())
    output = attn_weights.matmul(V)

    return output, attn_weights

# RoFormer: Enhanced Transformer with Rotary Position Embedding
# https://arxiv.org/abs/2104.09864
class RoPE:
    def __init__(self, hidden_size, rope_theta=10000):
        arr = Tensor.arange(0,hidden_size,2).div(hidden_size).unsqueeze(1)
        self.inv_freq = 1.0 / rope_theta ** arr

    def __call__(self, x: Tensor):
        B, H, S, M = x.shape
        pos =  Tensor.arange(0,S).unsqueeze(1)
        outer_product = pos.dot(self.inv_freq.transpose()) # similar to outer product


        output = x.empty_like()
        cos_theta = outer_product.cos()
        sin_theta = outer_product.sin()


        x_even = x[:,:,:,0:M:2]
        x_odd = x[:,:,:,1:M:2]

        out_x = x_even * cos_theta - x_odd * sin_theta
        out_y = x_even * sin_theta + x_odd * cos_theta

        output = output.stack(out_x,out_y, dim=-1).flatten(start_dim=-2)
        return output

# MQA: Fast Transformer Decoding: One Write-Head is All You Need https://arxiv.org/pdf/1911.02150
# MSA FOR 7B and 13B variants: Attention is All You Need https://arxiv.org/abs/1706.03762
# GQA FOR 34B and 70B variants : Training Generalized Multi-Query Transformer Models from Multi-Head https://arxiv.org/pdf/2305.13245
class Attention:
    def __init__(self, attention_bias, attention_dropout, head_dim, hidden_size, max_position_embeddings, num_attention_heads, num_key_value_heads, rope_theta, rms_norm_eps=1e-05):
        assert hidden_size == head_dim * num_attention_heads, "Hidden size does not match head dimension and number of attention heads"
        assert num_attention_heads % num_key_value_heads == 0, "Number of attention heads must be divisible by number of key value heads"
        self.g = num_attention_heads / num_key_value_heads 
        self.n_ah = num_attention_heads
        self.n_kvh = num_key_value_heads
        self.h_dim = head_dim * num_attention_heads
        self.q_proj = Linear(hidden_size,self.h_dim,attention_bias)
        self.k_proj = Linear(hidden_size,num_key_value_heads * head_dim,attention_bias)
        self.v_proj = Linear(hidden_size,num_key_value_heads * head_dim,attention_bias)
        self.o_proj = Linear(hidden_size,hidden_size, attention_bias)
        self.rope = RoPE(hidden_size,rope_theta)

    def __call__(self, x, mask=None):
        B, m, d_model = x.shape
        Q = self.q_proj(x).reshape(B, m, self.g, self.h_dim % self.g).permute(0,2,1,3)
        Q = self.rope(Q)

        K = self.k_proj(x).reshape(B, m, self.n_kvh, self.h_dim % self.n_kvh).permuate(0,2,1,3)
        K = self.rope(K)
        K = K.repeat_interleave(self.g,dim=1)
        
        V = self.v_proj(x).reshape(B, m, self.n_kvh, self.h_dim % self.n_kvh).permuate(0,2,1,3)
        V = V.repeat_interleave(self.g,dim=1)
        attn_score, attn_weights = scaled_dot_product_attn(Q,K,V,mask)
        concat_attn = attn_score.permute(0,2,1,3)reshape(B, m, self.h_dim * self.g)
        mhsa_output = self.o_proj(concat_attn)
        return mhsa_output,attn_weights

