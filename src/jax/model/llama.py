import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import Array
from typing import NamedTuple

from .embeddings import Embedding_params, create_embedding_params, embedding_forward
from .decoder import LlamaDecoder_params, create_llamadecoder_params, llamadecoder_forward

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

class Llama_params(NamedTuple): 
    embed_tokens: Embedding_params
    layer_params: List[LlamaDecoder_params]

def create_llama_params(attention_bias, head_dim, hidden_size, num_attention_heads, num_hidden_layers, num_key_value_heads, mlp_bias, vocab_size): -> LlamaDecoder_params:
    embed_tokens = create_embedding_params(vocab_size, hidden_size)
    layer_params = [create_llamadecoder_params(input_laynorm, mlp, post_attention_layernorm, self_attn)) for _ in range(num_hidden_layers)]

    return Llama_params(embed_tokens, layer_params)

def llama_forward(x: Array, params: Llama_params) -> Array:
    output = embedding_forward(x, params.embed_tokens)
    for idx,layer_param in enumerate(params.layer_params):
        output, attn_weights =  llamadecoder_forward(output, layer_param)
    return output

