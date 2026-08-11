# Pytorch's Implementation Llama, Llama2 and Llama3
import torch
from torch import nn
import torch.nn.functional as F
from .utils.model_utilities import scaled_dot_product_attention

# Llama Implementation https://github.com/meta-llama/llama/blob/main/llama/model.py?utm_source=chatgpt.com
# Recap Llama
# Tokenizer is SentencePiece's BytePairEncoding
# It is a Decoder Only Model
# No Learned/Absolute Embeddings, they directly use RoPe Embeddings
# Pre-Norm Instead of Post Norm
# Instead of GeLU they used SwiGLU (Gated Linear Unit)
# Implementation is in xformers, forward is inspired by the paper Self Attention Does Not Need O(n^2) Memory , backward Flashattention used
# Optimizer
# 6.7B 4096 32 32 3.0e−4 4M 1.0T

# Fix Dropout in Llama Later

# RoFormer: Enhanced Transformer with Rotary Postion Embedding https://arxiv.org/pdf/2104.09864
# Think it as applying rotation to every pair of dimensions
# Generating the inverse frequency, cos, sin better in fp32
# Apply the rope rotation can be done in fp16
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim=128, rope_theta=10000.0):
        super().__init__()

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x, start_pos=0):
        # x: [B, H, L, D]
        B, H, L, D = x.shape

        pos = torch.arange(
            start_pos,
            start_pos + L,
            device=x.device,
            dtype=self.inv_freq.dtype,
        )

        freqs = torch.outer(pos, self.inv_freq)  # [L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [L, D]

        cos = emb.cos()[None, None, :, :].to(dtype=x.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=x.dtype)

        return (x * cos) + (self.rotate_half(x) * sin)

# Root Mean Square Layer Normalization https://arxiv.org/pdf/1910.07467
# Main Finding is that Layer Normalization does re-centering and rescaling of invariance, but the main contribution is the scaling of invariance
# when computing the variance of hidden state, upcast for stability
class RMSNorm(nn.Module):
    def __init__(self, hidden_size=4096, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# For PaLM : Scaling Languaged Modeling with Paths ways https://arxiv.org/pdf/2204.02311
# d_ff = d_model * 4
# For Llama : Open and Efficient Language Model https://arxiv.org/pdf/2302.13971
# d_ff = d_model * 4 * 2 / 3
# GLU Variants Improve Transformer - https://arxiv.org/pdf/2002.05202
# Swish : A Self Gated Activation Function - https://arxiv.org/pdf/1710.05941v1
# Usually FFN : Linear -> Activation -> Linear
# But for FNN_SwiGLU this is not the case
# This can be in fully fp16
class FFNSwiGLU(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008):
        super().__init__()
        # We need Value and Gate Branch
        # SwiGLU = Swish(W2*x + A) * (W1*x + B)
        # W1 - value_branch, W2 - gate_branch
        self.gate_proj = nn.Linear(in_features=hidden_size, out_features=intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features=hidden_size, out_features=intermediate_size, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.down_proj = nn.Linear(in_features=intermediate_size, out_features=hidden_size,bias=False)
    def forward(self, x : torch.Tensor):
        # We can use F.silu(x) it is the same as swish(x)
        gate_output = self.gate_proj(x) 
        return self.down_proj(self.up_proj(x) * (gate_output * self.sigmoid(gate_output)))
        #return self.down_proj(self.up_proj(x) * F.silu(gate_output))

# Combining MSA, GQA and MQA into 1 module
# MQA: Fast Transformer Decoding: One Write-Head is All You Need https://arxiv.org/pdf/1911.02150
# Main idea: All the query head, share the same key and value head
# This is because the main bottleneck is the memory-bandwidth cost of repeatedly loading large key and value tensors.

# GQA:Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints https://arxiv.org/pdf/2305.13245
# 2 main idea to this GQA paper.
# 1. Converting MHA into GQA, which is called uptraining
# 2. Introduces Grouped-query attention which divides query heads into G groups, where each G groups shares 1 single k and v head. GQA-1 == MQA

# Projections, GEMM can all be in fp16
# softmax in scaled_dot_product_attention should be in fp32

# RoFormer: Enhanced Transformer with Rotary Position Embedding https://arxiv.org/pdf/2104.09864 RoPE is applied here
class Attention(nn.Module):
    def __init__(self, attention_bias, head_dim, hidden_size, max_position_embeddings, num_attention_heads, num_key_value_heads, rope_theta):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "Embedding dim is not divisible by number of heads"
        assert num_attention_heads % num_key_value_heads == 0, "Unable to obtain group side, n_heads not divisible"
        self.head_dim = hidden_size // num_attention_heads
        self.g = num_attention_heads // num_key_value_heads # Group Size
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.q_proj = nn.Linear(in_features=hidden_size, out_features=num_attention_heads * self.head_dim, bias=attention_bias) 
        self.k_proj = nn.Linear(in_features=hidden_size, out_features=num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(in_features=hidden_size, out_features=num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=attention_bias)
        self.rotary_emb = RotaryPositionEmbedding(head_dim = self.head_dim, rope_theta=rope_theta)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, start_pos=0, use_cache: bool = None):
        B, max_seq_len, d_model = x.shape
        Q = self.q_proj(x)
        Q = Q.reshape(B,max_seq_len,self.n_heads,self.head_dim).permute(0,2,1,3)
        
        K = self.k_proj(x)
        K = K.reshape(B,max_seq_len,self.n_kv_heads,self.head_dim).permute(0,2,1,3)

        # Apply RoPE to Q and K
        Q = self.rotary_emb(Q, start_pos)
        K = self.rotary_emb(K, start_pos)

        V = self.v_proj(x)
        V = V.reshape(B,max_seq_len,self.n_kv_heads,self.head_dim).permute(0,2,1,3)

        if use_cache:
            # Remember 2 phases Prefill and Decode
            if not hasattr(self, "kv_cache"):
                # Prefill 
                
                # Preallocate memory
                self.kv_cache = torch.empty(2, B, self.n_kv_heads, self.max_position_embeddings, self.head_dim, dtype=K.dtype, device=K.device)


            self.kv_cache[0, :, :, start_pos:start_pos+max_seq_len, :] = K               
            self.kv_cache[1, :, :, start_pos:start_pos+max_seq_len, :] = V               

            K = self.kv_cache[0, :, :, :start_pos+max_seq_len, :]
            V = self.kv_cache[1, :, :, :start_pos+max_seq_len, :]

        # 1. If we use torch.repeat_interleave does it really allows us the benefit of whatever mqa proposed which is less memory access of K and V Tensors
        # 2. Does flash_attention adjust to MQA,GQA or is it strictly MSA only
        # 3. What is the usual format for allow the Attention module to take start_pos as well as input tokens (full batch for training)

        K = torch.repeat_interleave(K,repeats = self.g, dim=1)
        V = torch.repeat_interleave(V,repeats = self.g, dim=1)

        mhsa_output,attn_weights = scaled_dot_product_attention(Q=Q,K=K,V=V,mask=mask)
        concat_output = mhsa_output.permute(0,2,1,3).reshape(B,max_seq_len, d_model)
        output = self.o_proj(concat_output) 

        return output, attn_weights

# Llama: Open and Efficient Foundation Model: https://arxiv.org/pdf/2302.13971
# Architecture is the same as the decoder used in the "Attention is all you need"
# PreNorm Used instead of PostNorm, but residual streams should stay unnormalized
# LLAMA 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/pdf/2307.09288
# Similar Architecturally to Llama, only difference is GQA instead of MHA
# PreNorm, RoPE, SwiGLU 
# Decoder Only
# Tokenizer still Byte Pair Encoding implemented by SentencePiece
# GQA is only applied to 34B and 70B models
class LlamaDecoder(nn.Module):
    def __init__(self, attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_key_value_heads, mlp_bias, rope_theta):
        super().__init__()
        self.mlp = FFNSwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.self_attn = Attention(attention_bias=attention_bias, head_dim=head_dim, hidden_size=hidden_size, max_position_embeddings=max_position_embeddings, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, rope_theta=rope_theta)
        self.input_layernorm = RMSNorm(hidden_size=hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, mask:torch.Tensor = None, start_pos=0, use_cache=False):
        norm_x = self.input_layernorm(x)
        self_attn_output, attn_weights = self.self_attn(norm_x, mask=mask, start_pos=start_pos, use_cache=use_cache)
        rms2_input = self_attn_output + x
        ffn_input = self.post_attention_layernorm(rms2_input)
        ffn_output = self.mlp(ffn_input)
        output = ffn_output + rms2_input
        return output
    
# For Training initializer_range: float = 0.02
class Llama(nn.Module):
    def __init__(self, attention_bias = False, attention_dropout=0.0, head_dim=128, hidden_size = 4096, intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32, num_key_value_heads = 32, max_position_embeddings = 4096, mlp_bias=False, rope_theta=10000, vocab_size=32000):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, max_position_embeddings, num_attention_heads, num_key_value_heads, mlp_bias, rope_theta) for _ in range(num_hidden_layers)])
        self.lm_head = nn.Linear(in_features=hidden_size,out_features=vocab_size, bias=False)
        self.norm = RMSNorm(hidden_size=hidden_size)
    
    def forward(self, input_ids : torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor, start_pos=0, use_cache=False):
        output = self.embed_tokens(input_ids)
        for idx,layer in enumerate(self.layers):
            output = layer(output, mask=attention_mask, start_pos=start_pos, use_cache=use_cache)

        output = self.norm(output)
        output = self.lm_head(output)
        return output
