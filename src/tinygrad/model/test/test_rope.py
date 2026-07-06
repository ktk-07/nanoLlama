import tinygrad
from tinygrad import Tensor

class RoPE:
    def __init__(self, hidden_size, rope_theta=10000):
        arr = Tensor.arange(0,hidden_size,2).div(hidden_size).unsqueeze(1)
        self.inv_freq = (1.0 / rope_theta ** arr).is_param_(False)
        
    def __call__(self, x: Tensor):
        B, H, S, M = x.shape
        pos =  Tensor.arange(0,S).unsqueeze(1) 
        outer_product = pos.dot(self.inv_freq.transpose()) # similar to outer product

        output = x.empty_like()
        cos_theta = outer_product.cos()
        sin_theta = outer_product.sin()

        x_even = x[:,:,:,0:M:2]
        x_odd = x[:,:,:,1:M:2]

        out_x = x_even * cos_theta - x_odd * sin_theta
        out_y = x_even * sin_theta + x_odd * cos_theta

        output = output.stack(out_x,out_y, dim=-1).flatten(start_dim=-2)
        return output

B = 1
H = 1
S = 4
M = 2


x = Tensor.randn(B,H,S,M)
y = Tensor.randn(B,H,S,M)

output = Tensor.empty(B,H,S*2,M)
print(x.numpy())
print(x.numpy().shape)
print(y.numpy())
print(y.numpy().shape)

print(Tensor.stack(x,y,dim=-1).numpy())
print(Tensor.stack(x,y,dim=-1).numpy().shape)

print(Tensor.stack(x,y,dim=-1).flatten(start_dim=-2).numpy())
print(Tensor.stack(x,y,dim=-1).flatten(start_dim=-2).numpy().shape)
