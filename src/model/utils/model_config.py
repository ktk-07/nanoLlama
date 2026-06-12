@dataclass
class LlamaConfig:
    attention_bias: bool
    attention_dropout: bool
    head_dim: int
    hidden_size: int
    initializer_range: float = 0.02
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings : int
    intermediate_size: int
    rope_theta: float = 10000.0
    rope_type: str = "default"    
    mlp_bias: bool
    vocab_size: int

