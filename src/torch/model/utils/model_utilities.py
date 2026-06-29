import torch
from torch import nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask:torch.Tensor = None):
    # Matmul in fp16
    B, h, max_seq_len, d_k = Q.shape
    attn_logits = Q @ K.transpose(-1,-2) / math.sqrt(d_k)

    # Upcast before Softmax in fp32
    attn_logits = attn_logits.float()
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask,torch.finfo(attn_logits.dtype).min)
    
    # Softmax in fp32 then cast back to fp16
    attn_weights = attn_logits.softmax(dim=-1).to(dtype=V.dtype)

    # Matmul in fp16
    attn_score =  attn_weights @ V
    return (attn_score, attn_weights)
