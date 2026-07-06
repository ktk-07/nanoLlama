import tinygrad
from tinygrad import Tensor
import tinygrad.nn as nn
import math


#
class Linear:
    def __init__(self, in_features : int, out_features : int, bias : bool = False):
        pass
    def __call__(self):
        pass

# Recap LLama 
class RotaryPositionEmbeddings:
     def __init__(self,):
        self.inv_freq = n;
        self.
    def __call__(self):
        pass

class RMSNorm:
     def __init__(self,):
        self.weight = 
        self.scale = 
    def __call__(self):
        pass


class MLP:
     def __init__(self,):
        self.
        pass
    def __call__(self):
        pass

# MQA: Fast Transformer Decoding: One Write-Head is All You Need https://arxiv.org/pdf/1911.02150
# MSA FOR 7B and 13B variants: Attention is All You Need https://arxiv.org/abs/1706.03762
# GQA FOR 34B and 70B variants : Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints https://arxiv.org/pdf/2305.13245
def scaled_dot_product_attn(Q:Tensor, K:Tensor, V:Tensor):

    return attn_score ,attn_weights

class Attention:
    def __init__(self,):
        pass
    def __call__(self):
        pass

class LLama2Decoder:
     def __init__(self,):
        self.attn = Attention()
        self. = RMSNorm()
        self. = RMSNorm()
        pass
    def __call__(self):
        pass

# LLAMA 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/pdf/2307.09288
class LLama2:
     def __init__(self,):
        pass
    def __call__(self):
        pass


