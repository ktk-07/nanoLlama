import tinygrad
from tinygrad import Tensor

class Linear:
    def __init__(self, in_features, out_features, bias=False):
        self.W = Tensor.randn(in_features,out_features)
        if bias:
            self.b = Tensor.randn(out_features)
        else:
            self.b = None

    def __call__(self, x: Tensor):
        if self.b is Tensor:
            return x.matmul(self.W) + self.b
        return x.matmul(self.W) 

class MLP:
    def __init__(self, hidden_size, intermediate_size, mlp_bias=False):
        self.gate_proj = Linear(hidden_size, intermediate_size, mlp_bias)
        self.up_proj = Linear(hidden_size, intermediate_size, mlp_bias)
        self.down_proj = Linear(intermediate_size, hidden_size, mlp_bias)
    def __call__(self, x):
        gate_proj_output = self.gate_proj(x)
        up_proj_output = self.up_proj(x)
        down_proj_input = up_proj_output.mul(gate_proj_output * gate_proj_output.sigmoid())
        output = self.down_proj(down_proj_input)
        return output


x = Tensor.arange(3*2).reshape(3, 2)
hidden_size = 2
intermediate_size = 4
mlp = MLP(hidden_size,intermediate_size)

print(mlp(x).numpy())
print(mlp(x).shape)
