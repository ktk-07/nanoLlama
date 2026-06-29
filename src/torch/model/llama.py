# Pytorch's Implementation Llama, Llama2 and Llama3
import torch
from torch import nn
import torch.nn.functional as F
from .utils.model_utilities import scaled_dot_product_attention

# I have not integrated this the models with KV-caching

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


# We can change the code so that it accomodates MHA, MQA, GQA
# RoFormer: Enhanced Transformer with Rotary Position Embedding https://arxiv.org/pdf/2104.09864 RoPE is applied here
class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_attention_heads = 32, hidden_size = 4096, d_k = 4096):
        super().__init__()
        assert d_model % h == 0 , f"Hidden Dimensions {d_model} not divisible by num of heads {h}"
        self.h = num_attention_heads
        self.d_model = hidden_size
        self.head_dim = self.d_model // self.h
        self.linear_q = nn.Linear(in_features=self.d_model, out_features=d_k)
        self.linear_k = nn.Linear(in_features=self.d_model, out_features=d_k)
        self.linear_v = nn.Linear(in_features=self.d_model, out_features=d_k)
        self.linear_o = nn.Linear(in_features=d_k, out_features=self.d_model)
        self.rope = RotaryPositionEmbedding(head_dim=self.head_dim, base=10000)

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        B, max_seq_len, d_model = Q.shape
        assert d_model % self.h == 0 , f"Hidden Dimensions {d_model} not divisible by num of heads {self.h}"

        Q = self.linear_q(x)
        Q = Q.reshape(B, max_seq_len, self.h, d_model // self.h).permute(0,2,1,3)
        K = self.linear_q(x)
        K = K.reshape(B, max_seq_len, self.h, d_model // self.h).permute(0,2,1,3)
        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)

        V = self.linear_q(x)
        V = V.reshape(B, max_seq_len, self.h, d_model // self.h).permute(0,2,1,3)

        attn_score, attn_weights = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attn_score_reshaped = attn_score.permute(0,2,1,3).reshape(B, max_seq_len, d_model)
        mhsa_output = self.linear_o(attn_score_reshaped)

        return mhsa_output, attn_weights

# MQA: Fast Transformer Decoding: One Write-Head is All You Need https://arxiv.org/pdf/1911.02150
# Main idea: All the query head, share the same key and value head
# This is because the main bottleneck is the memory-bandwidth cost of repeatedly loading large key and value tensors.

class MQA(nn.Module):
    def __init__(self, num_attention_heads=32, num_key_value_heads=1, hidden_size=4096):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "Embedding Dimension must be divisible by H"
        assert num_attention_heads % num_key_value_heads == 0, "Total Number of heads must be divisible by number of group"
        self.h = num_attention_heads # Total no of heads
        self.g = self.h // num_key_value_heads # Group Size
        self.head_dim = hidden_size // self.h # Head Dimension
        self.num_of_heads = self.h // self.g # Number of heads in a group
        self.linear_q = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.linear_k = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.linear_v = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.linear_o = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.rope = RotaryPositionEmbedding(head_dim=self.head_dim, base=10000)

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        # input : batch x max_seq_len x d_model
        B, max_seq_len, d_model = x.shape
        Q = self.linear_q(x)
        Q = Q.reshape(B, max_seq_len, self.h, self.head_dim).permute(0,2,1,3) # batch_size x num_of_heas x max_seq_len x self.head_dim
        # The reason why we do this is because we need each query head to be independent

        K = self.linear_k(x) # batch x max_seq_len x head_dim
        K = K.reshape(B, max_seq_len, self.g, self.head_dim).permute(0,2,1,3) # batch_size x num_of_groups x max_seq_len x self.head_dim

        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)

        V = self.linear_v(x) # batch x max_seq_len x head_dim
        V = V.reshape(B, max_seq_len, self.g, self.head_dim).permute(0,2,1,3) # batch_size x num_of_groups x max_seq_len x self.head_dim

        # Important to use repeat_interleave instead of repeat to match group strucutre
        K = K.repeat_interleave(self.num_of_heads,dim=1) # B x self.h x max_seq_len x self.head_dim
        V = V.repeat_interleave(self.num_of_heads,dim=1) # B x self.h x max_seq_len x self.head_dim

        # input B x self.h x max_seq_len x self.head_dim
        attn_score, attn_weights = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attn_score_reshaped = attn_score.permute(0,2,1,3).reshape(B, max_seq_len, d_model)
        mhsa_output = self.linear_o(attn_score_reshaped)
        return mhsa_output, attn_weights

# GQA:Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints https://arxiv.org/pdf/2305.13245
# 2 main idea to this GQA paper.
# 1. Converting MHA into GQA, which is called uptraining
# 2. Introduces Grouped-query attention which divides query heads into G groups, where each G groups shares 1 single k and v head. GQA-1 == MQA
class GroupQueryAttention(nn.Module):
    def __init__(self, head_dim= 128, hidden_size=4096, num_attention_heads=32, num_key_value_heads=1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "Embedding Dimension must be divisible by H"
        assert num_attention_heads % num_key_value_heads == 0, "Total Number of heads must be divisible by number of group"
        self.h = num_attention_heads # Total no of heads
        self.g = self.h // num_key_value_heads # Group Size
        self.head_dim = hidden_size // self.h # Head Dimension
        self.num_of_heads = self.h // self.g # Number of heads in a group
        self.q_proj = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.k_proj = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.v_proj = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.o_proj = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.rope = RotaryPositionEmbedding( head_dim=self.head_dim, base=10000)

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        # input : batch x max_seq_len x d_model
        B, max_seq_len, d_model = x.shape
        Q = self.linear_q(x)
        Q = Q.reshape(B, max_seq_len, self.h, self.head_dim).permute(0,2,1,3) # batch_size x num_of_heas x max_seq_len x self.head_dim
        # The reason why we do this is because we need each query head to be independent

        K = self.linear_k(x) # batch x max_seq_len x head_dim
        K = K.reshape(B, max_seq_len, self.g, self.head_dim).permute(0,2,1,3) # batch_size x num_of_groups x max_seq_len x self.head_dim

        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)

        V = self.linear_v(x) # batch x max_seq_len x head_dim
        V = V.reshape(B, max_seq_len, self.g, self.head_dim).permute(0,2,1,3) # batch_size x num_of_groups x max_seq_len x self.head_dim

        # Important to use repeat_interleave instead of repeat to match group strucutre
        K = K.repeat_interleave(self.num_of_heads,dim=1) # B x self.h x max_seq_len x self.head_dim
        V = V.repeat_interleave(self.num_of_heads,dim=1) # B x self.h x max_seq_len x self.head_dim

        # input B x self.h x max_seq_len x self.head_dim
        attn_score, attn_weights = scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attn_score_reshaped = attn_score.permute(0,2,1,3).reshape(B, max_seq_len, d_model)
        mhsa_output = self.linear_o(attn_score_reshaped)
        return mhsa_output, attn_weights

# Combining MSA, GQA and MQA into 1 module
# Projections, GEMM can all be in fp16
# softmax in scaled_dot_product_attention should be in fp32
class Attention(nn.Module):
    def __init__(self, attention_bias, head_dim, hidden_size, num_attention_heads, num_key_value_heads, rope_theta):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "Embedding dim is not divisible by number of heads"
        assert num_key_value_heads % num_key_value_heads == 0, "Unable to obtain group side, n_heads not divisible"
        self.head_dim = hidden_size // num_attention_heads
        self.g = num_attention_heads // num_key_value_heads # Group Size
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.q_proj = nn.Linear(in_features=hidden_size, out_features=num_attention_heads * self.head_dim, bias=attention_bias) 
        self.k_proj = nn.Linear(in_features=hidden_size, out_features=num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(in_features=hidden_size, out_features=num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=attention_bias)
        self.rotary_emb = RotaryPositionEmbedding(head_dim = self.head_dim, rope_theta=rope_theta)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, max_seq_len, d_model = x.shape
        Q = self.q_proj(x)
        Q = Q.reshape(B,max_seq_len,self.n_heads,self.head_dim).permute(0,2,1,3)
        
        K = self.k_proj(x)
        K = K.reshape(B,max_seq_len,self.n_kv_heads,self.head_dim).permute(0,2,1,3)

        # Apply RoPE to Q and K
        Q = self.rotary_emb(Q)
        K = self.rotary_emb(K)

        V = self.v_proj(x)
        V = V.reshape(B,max_seq_len,self.n_kv_heads,self.head_dim).permute(0,2,1,3)

        K = torch.repeat_interleave(K,repeats = self.g, dim=1)
        V = torch.repeat_interleave(V,repeats = self.g, dim=1)

        mhsa_output,attn_weights = scaled_dot_product_attention(Q=Q,K=K,V=V,mask=mask)
        concat_output = mhsa_output.permute(0,2,1,3).reshape(B,max_seq_len, d_model)
        output = self.o_proj(concat_output) 

        return output, attn_weights

# Attention is all you need https://arxiv.org/abs/1706.03762
# Architecture is the same as the decoder used in the original attention paper
# PreNorm Used instead of PostNorm, but residual streams should stay unnormalized
class LlamaDecoder(nn.Module):
    def __init__(self, attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, max_position_embeddings, rope_theta):
        super().__init__()
        self.mlp = FFNSwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.self_attn = MQA(attention_bias=attention_bias, head_dim=head_dim, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, rope_theta=rope_theta)
        self.input_layernorm = RMSNorm(hidden_size=hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size=hidden_size)

    def forward(self, x : torch.Tensor, attention_mask: torch.Tensor):
        norm_x = self.input_layernorm(x)
        mhsa_output, attn_weights = self.self_attn(norm_x, mask=attention_mask)
        rms2_input = mhsa_output + x
        ffn_input = self.post_attention_layernorm(rms2_input)
        ffn_output = self.mlp(ffn_input)
        output = ffn_output + rms2_input
        return output

class Llama(nn.Module):
    def __init__(self, attention_bias = False, attention_dropout=0.0, head_dim=128, hidden_size = 4096, intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32, num_key_value_heads = 32, mlp_bias=False, rope_theta=10000, vocab_size=32000):
        super().__init__()
        self.tkn_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, mlp_bias, rope_theta) for _ in range(num_hidden_layers)])
        self.lm_head = nn.Linear(in_features=hidden_size,out_features=vocab_size, bias=False)

    def forward(self, x :torch.Tensor, mask : torch.Tensor = None):
        output = self.tkn_embedding(x)
        for idx,layer in enumerate(self.layers):
            output = layer(output, mask=mask)

        output = self.lm_head(output)
        return output

# LLAMA 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/pdf/2307.09288
# Similar Architecturally to Llama, only difference is GQA instead of MHA
# PreNorm, RoPE, SwiGLU 
# Decoder Only
# Tokenizer still Byte Pair Encoding implemented by SentencePiece
# GQA is only applied to 34B and 70B models
class Llama2Decoder(nn.Module):
    def __init__(self, attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, mlp_bias, rope_theta):
        super().__init__()
        self.mlp = FFNSwiGLU(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.self_attn = Attention(attention_bias=attention_bias, head_dim=head_dim, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads, rope_theta=rope_theta)
        self.input_layernorm = RMSNorm(hidden_size=hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, mask:torch.Tensor = None):
        norm_x = self.input_layernorm(x)
        self_attn_output, attn_weights = self.self_attn(norm_x, mask=mask)
        rms2_input = self_attn_output + x
        ffn_input = self.post_attention_layernorm(rms2_input)
        ffn_output = self.mlp(ffn_input)
        output = ffn_output + rms2_input
        return output
    
# For Training initializer_range: float = 0.02
class Llama2(nn.Module):
    def __init__(self, attention_bias = False, attention_dropout=0.0, head_dim=128, hidden_size = 4096, intermediate_size = 11008, num_hidden_layers = 32, num_attention_heads = 32, num_key_value_heads = 32, mlp_bias=False, rope_theta=10000, vocab_size=32000):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size)
        self.layers = nn.ModuleList([Llama2Decoder(attention_bias, attention_dropout, head_dim, hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, mlp_bias, rope_theta) for _ in range(num_hidden_layers)])
        self.lm_head = nn.Linear(in_features=hidden_size,out_features=vocab_size, bias=False)
        self.norm = RMSNorm(hidden_size=hidden_size)
    
    def forward(self, input_ids : torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor):
        output = self.embed_tokens(input_ids)
        for idx,layer in enumerate(self.layers):
            output = layer(output, mask=attention_mask)

        output = self.norm(output)
        output = self.lm_head(output)
        return output
