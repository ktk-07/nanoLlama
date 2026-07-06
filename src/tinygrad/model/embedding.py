import tinygrad
from tinygrad import Tensor

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor.randn(num_embeddings, embedding_dim)
    def __call__(self, token_id):
        return self.weight[token_id]
