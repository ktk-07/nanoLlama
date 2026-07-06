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


batch = 2
seq_len = 4
vocab_size = 1000
embedding_dim = 64

params = create_embedding_params(vocab_size, embedding_dim)

x = jnp.array([
    [5, 12, 99, 7],
    [3, 3, 18, 42],
])

out = embedding_forward(x, params)

print(out.shape)
# (2, 4, 64)
