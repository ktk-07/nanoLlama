import jax
import jax.numpy as jnp
from jax import Array
from typing import NamedTuple

key = jax.random.key(42)
class RMSNorm_params(NamedTuple):
    weight: Array
    eps: float

def create_rmsnorm_params(hidden_size, eps=1e-6) -> RMSNorm_params:
    weight = jnp.ones((hidden_size))
    return  RMSNorm_params(weight, eps)

def rmsnorm_forward(x: Array, params: RMSNorm_params) -> Array:
    ms = jnp.mean(x**2, axis=-1, keepdims=True) + params.eps
    return x / jnp.sqrt(ms) * params.weight
