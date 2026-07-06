import jax
from jax import Array
import jax.numpy as jnp
from typing import NamedTuple

key = jax.random.key(42)

class Embedding_params(NamedTuple):
    weight: Array

def create_embedding_params(num_embeddings, embedding_dim) -> Embedding_params:
    weight = jax.random.normal(key, (num_embeddings, embedding_dim))
    return Embedding_params(weight)

def embedding_forward(x: Array, params:Embedding_params) -> Array:
    return params.weight[x]

