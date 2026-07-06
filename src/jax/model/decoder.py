import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import Array
from typing import NamedTuple

from .mlp import MLP_params, create_mlp_params, mlp_forward
from .attn import Attention_params, create_attn_params, attn_forward 
from .rms_norm import RMSNorm_params, create_rmsnorm_params, rmsnorm_forward

class LlamaDecoder_params(NamedTuple):
    input_layernorm : RMSNorm_params
    mlp : MLP_params
    post_attention_layernorm : RMSNorm_params
    self_attn : Attention_params

def create_llamadecoder_params(attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads, mlp_bias): -> LlamaDecoder_params:
    input_laynorm = create_rmsnorm_params(hidden_size)
    mlp : MLP_params = create_mlp_params(hidden_size, intermediate_size, mlp_bias)
    post_attention_layernorm = create_rmsnorm_params(hidden_size)
    self_attn = create_attn_params(attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads)

    return LlamaDecoder_params(input_laynorm, mlp, post_attention_layernorm, self_attn)

def llamadecoder_forward(x: Array, params: LlamaDecoder_params) -> Array:
    norm_x = rmsnorm_forward(x, params.input_layernorm)
    attn_output, attn_weight = attn_forward(norm_x, params.self_attn)
    post_attn_input = attn_output + x
    norm_post_attn_input = rmsnorm_forward(x, params.post_attention_layernorm)
    output = mlp_forward(norm_post_attn_input, params.mlp)

    return output + attn_output

