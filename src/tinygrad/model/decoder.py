import tinygrad
from tinygrad import Tensor

from .rmsnorm import RMSNorm
from .attn import Attention
from .mlp import MLP

class LlamaDecoder:
    def __init__(self,attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads, mlp_bias):
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size,intermediate_size)
        self.self_attn = Attention(attention_bias, attention_dropout, head_dim, hidden_size, max_position_embeddings, num_attention_heads, num_key_value_heads, rope_theta, rms_norm_eps=1e-05)

    def __call__(self, x: Tensor, mask: Tensor = None) -> Tensor:
        norm_x = self.input_layer_norm(x)
        attn_output, attn_weight = self.attn(norm_x, mask)
        post_attn_input = attn_output + x
        norm_post_attn_input = self.post_attention_layernorm(x) 
        output = self.mlp(norm_post_attn_input)
        return output + attn_output
