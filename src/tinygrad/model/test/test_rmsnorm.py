import tinygrad
from tinygrad import Tensor

class RMSNorm:
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = Tensor.randn(hidden_size)
        self.eps = eps
    def __call__(self, x : Tensor):
        mean = (x**2).mean(axis=-1, keepdim = True)
        # return x / ( (x.sum(axis=-1, keepdim=True).div(hidden_size))**2 + eps).sqrt())
        return x / math.sqrt(mean + self.eps)

Tensor.manual_seed(69)
batch = 1
seq_len = 6
hidden_size = 9
eps = 1e-6
x = Tensor.randn(batch, seq_len, hidden_size)

out1 = (x / ((x**2).mean(axis=-1, keepdim=True) + eps).sqrt())
out2 = (x / ( ((x**2).sum(axis=-1, keepdim=True).div(hidden_size)) + eps).sqrt())

print(Tensor.allclose(out1,out2).numpy())
