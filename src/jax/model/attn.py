import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import Array
from typing import NamedTuple
import math
from .
key = jax.random.key(42)
class Linear_params(NamedTuple):
    W: Array
    b: Array | None

def create_linear_params(in_features, out_features, bias=False) -> Linear_params:
    W = jax.random.normal(key,(in_features,out_features))
    b = None
    if bias:
        b = jax.random.normal(key,(out_features))
    params = Linear_params(W,b)
    return params

def linear_forward(x : Array, linear_params: Linear_params):
    assert x.shape[-1] == linear_params.W.shape[0], f"Shape mismatchi matmul {x.shape} {linear_params.W.shape}"
    if linear_params.b is None:
        return x.dot(linear_params.W)
    return x.dot(linear_params.W) + linear_params.b

def create_inv_freq(hidden_size, rope_theta=10000):
    inv_freq = 1 / (rope_theta ** (jnp.arange(0,hidden_size,2) / hidden_size))
    return inv_freq 

def apply_rope(x : Array, inv_freq: Array) -> Array:
    B, H, S, M = x.shape
    pos = jnp.arange(0,S)
    theta = jnp.outer(pos,inv_freq) 

    x_even = [...,:M:2]
    x_odd = [...,1:M:2]

    cos = jnp.cos(theta)
    sin = jnp.sin(theta)

    # apply half rotation
    output1 = x_even * cos - x_odd * sin
    output2 = x_even * cos - x_odd * sin
    
    return (jax.stack([output1,output2],axis-2).ravel(axis=-1)

def scaled_dot_product_attn(Q : Array, K : Array, V : Array, mask:Array = None):
    B, H, S, d_k = x.shape
    attn_logits = (Q @ jnp.swapaxes(K,-1,-2))  / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -1e9)
    
    attn_weights = jnn.softmax(attn_logits, axis=-1, initial=0.0)
    attn_score = attn_weights @ V

    return attn_score, attn_weights

class Attention_params(NamedTuple):
    g : int
    head_dim:int
    num_attention_heads: int
    num_key_value_heads: int
    q_proj : Linear_params
    k_proj : Linear_params
    v_proj : Linear_params
    o_proj : Linear_params    
    inv_freq : Array # Need to turn off grad for this

def create_attn_params(attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads) -> Linear_params:
    assert num_attention_heads % num_key_value_heads == 0, "Must be divisible!"
    g = num_attention_heads / num_key_value_heads
    q_proj = create_linear_params(hidden_size, head_dim * num_attention_heads, attention_bias)    
    k_proj = create_linear_params(hidden_size, head_dim * num_key_value_heads, attention_bias)    
    v_proj = create_linear_params(hidden_size, head_dim * num_key_value_heads, attention_bias)    
    o_proj = create_linear_params(hidden_size, hidden_size, attention_bias)    
    inv_freq = create_inv_freq(hidden_size)

    return Attention_params(g, head_dim, num_attention_heads, num_key_value_heads, q_proj, k_proj, v_proj, o_proj, inv_freq)

def attn_forward(x : Array, mask:None, params: Linear_params):
    B, S, H = x.shape

    Q = linear_forward(x , params.q_proj)
    Q = Q.reshape(B, S, params.num_attention_heads, head_dim).tranpose(0,2,1,3)
    Q = apply_rope(Q, params.inv_freq)

    K = linear_forward(x , params.k_proj)
    K = K.reshape(B, S, params.num_key_values_heads, head_dim).tranpose(0,2,1,3)
    K = apply_rope(K)
    K = jnp.repeat(K, params.inv_freq)

    V = linear_forward(x , params.v_proj)
    V = V.reshape(B, S, params.num_key_values_heads, head_dim).tranpose(0,2,1,3)
    V = apply_rope(V)
    V = jnp.repeat(V, params.inv_freq)

    attn_output, attn_weights = scaled_dot_prodcut_attn(Q,K,V,mask)
    concat_attn = attn_output.permute(0,2,1,3).reshape(B, S, H)
    mhsa_output = linear_forward(concat_attn, params.o_proj)
    
    return mhsa_output, attn_weights








