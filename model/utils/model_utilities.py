import torch
from torch import nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q : torch.Tensor, K : torch.Tensor, V : torch.Tensor, mask:torch.Tensor = None):
    B, h, max_seq_len, d_k = Q.shape
    attn_weights = Q @ K.transpose(-1,-2) / math.sqrt(d_k)
    if mask is not none:
        attn_weights = attn_weights.masked_fill(mask,float("-inf"))
    attn_weights = attn_weights.softmax(dim=-1)
    attn_score =  attn_weights @ V
    return (attn_score, attn_weights)

def precompute_rotation_matrix(d_model):
    # I have a question: When is the position embedding computed? before every forward pass or no?
    # The problem is that we may have variable seq len no?

    # What if d_model % 2 == 0
    # 1.generate position matrix?
    # 2.generate
    r_theta = 0 

    pos_matrix = torch.arange(0,512).unsqueeze(0)
    #pos_matrix = pos_matrix.repeat(512,dim=0) # actually dont need this it will be broadcasted anyways
    # Generate Rotaton Matrix

    # The efficient trick is to use elementwise cos/sin tables plus a rotate_half operation
    # Usually rotation is applied like this Rv  R = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]  v = [x,y]
    # Rv = [[x*cos_theta - y*sin_theta], [x*sin_theta + y*cos_theta]]
    print(pos_matrix)

    return r_theta


def precompute_theta_for_rope(d_model, base=10000):
    output = torch.arange(0,d_model,2)
    output = -output / d_model * torch.log(base)
    return output.unsqueeze(0)
