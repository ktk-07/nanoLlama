import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from typing import NamedTuple
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


# Test Linear
x = jax.random.normal(key,(1,10))
w = jax.random.normal(key,(10,7))
b = jax.random.normal(key,(7))
print(x.dot(w) + (b))

params = create_linear_params(10,7,True)
print(params)
output = linear_forward(x,params)
print(output)


params = create_linear_params(10,7,False)
print(params)
output = linear_forward(x,params)
print(output)

class MLP_params(NamedTuple):
    up_proj_params : Linear_params
    gate_proj_params : Linear_params
    down_proj_params : Linear_params

def create_mlp_params(hidden_size, intermediate_size, mlp_bias) -> MLP_params:
    up_proj_params = create_linear_params(hidden_size, intermediate_size, mlp_bias)
    gate_proj_params = create_linear_params(hidden_size, intermediate_size, mlp_bias)
    down_proj_params = create_linear_params(intermediate_size, hidden_size, mlp_bias)

    return MLP_params(up_proj_params, gate_proj_params, down_proj_params)

def mlp_forward(x: Array, params:MLP_params) -> Array:
    up_proj_output = linear_forward(x, params.up_proj_params)
    gate_proj_output = linear_forward(x, params.gate_proj_params)
    act_output = gate_proj_output * jax.nn.sigmoid(gate_proj_output) * up_proj_output 
    output = linear_forward(act_output, params.down_proj_params)    
    return output

# Test MLP with Swiglu

hidden_size = 4096
intermediate_size = 11008
mlp_bias = False

x = jax.random.normal(key, (1,hidden_size))
mlp_params = create_mlp_params(hidden_size, intermediate_size, mlp_bias)
output = mlp_forward(x, mlp_params)

print(output.shape)



