import tinygrad
from tinygrad import Tensor
import tinygrad.nn as nn
import math

from .embeddings import Embedding
from .decoder import LlamaDecoder

@dataclass
class LlamaConfig:
  attention_bias:bool = false
  attention_dropout:float =  0.0
  dtype:str = "float16"
  head_dim:int = 128
  hidden_size:int = 4096
  intermediate_size:int = 11008
  max_position_embeddings:int = 4096
  mlp_bias: bool = False,
  num_attention_heads:int = 32
  num_hidden_layers:int = 32
  num_key_value_heads: int = 32
  rms_norm_eps: float = 1e-05
  rope_theta: float = 10000.0
  use_cache: bool = True
  vocab_size: int =  32000

# LLAMA 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/pdf/2307.09288
class LLama2:
     def __init__(self, attention_bias, head_dim, hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, mlp_bias, vocab_size):
        self.embed_tokens = Embedding(vocab_size, hidden_size)
        self.layers = [LlamaDecoder(attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads, mlp_bias) for _ in range(num_hidden_layers)]

    def __call__(self, x, mask):
        output = self.embed_tokens(x);
        for idx,layer in enumerate(self.layers):
            output, attn_weights = layer(output,mask)
            
        return output


