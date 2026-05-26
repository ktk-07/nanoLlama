# Pytorch's Implementation Llama, Llama2 and Llama3
import torch
from torch import nn
import torch.nn.function as F
from .utils import scaled_dot_product_attention, generate_matrixes_for_rope

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
def generate_theta_for_rope(d_model, base=10000):
    output = torch.arange(0,d_model,2)
    output = -output / d_model * torch.log(10000)
    return output.float()

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, head_dim, base=10000):
        assert head_dim % 2 == 0 , "Head Dimension must be divisible by 2"
        self.max_seq_len = max_seq_len 
        self.head_dim = head_dim
        inv_freq = generate_matrixes_for_rope(head_dim, base=base)
        pos = torch.arange(max_seq_len).float()
        theta = torch.outer(pos,inv_freq)
        cos = torch.cos(pos * torch.exp(theta))
        sin = torch.sin(pos * torch.exp(theta))

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, val : torch.Tensor, ):
        # val batch_size x h x max_seq_len x d_model

        B, max_seq_len, d_model = val.shape
        x = val[:, :, 0::2]
        y = val[:, :, 0::2]

        xcos_t = x * self.cos
        xsin_t = x * self.sin
        ycos_t = y * self.cos
        ysin_t = y * self.sin

        actual_x = xcos_t - ysin_t
        actual_y = xsin_t + ycos_t 

        output = torch.empty_like(val)
        output[:, 0::2, :] = actual_x
        output[:, 1::2, :] = actual_y  
        return output

# Root Mean Square Layer Normalization https://arxiv.org/pdf/1910.07467
class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,d_model))
        self.eps = 1e-7
    def forward(self, x : torch.Tensor):
        rms_val = x.pow(2).mean(dim=-1,keep_dim=True)
        output = x / torch.sqrt(rms_val + self.eps)
        return output * self.gamma

# For PaLM : Scaling Languaged Modeling with Paths ways https://arxiv.org/pdf/2204.02311
# d_ff = d_model * 4
# For Llama : Open and Efficient Language Model https://arxiv.org/pdf/2302.13971
# d_ff = d_model * 4 * 2 / 3
# GLU Variants Improve Transformer - https://arxiv.org/pdf/2002.05202
# Swish : A Self Gated Activation Function - https://arxiv.org/pdf/1710.05941v1
# Usually FFN : Linear -> Activation -> Linear
# But for FNN_SwiGLU this is not the case
class FFNSwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # We need Value and Gate Branch
        # SwiGLU = Swish(W2*x + A) * (W1*x + B)
        # W1 - value_branch, W2 - gate_branch
        self.value_branch = nn.Linear(d_model, d_ff)
        self.gate_branch = nn.Linear(d_model, d_ff) 
        self.sigmoid = nn.Sigmoid()
        self.w = nn.Linear(d_ff,d_model)
    def forward(self, x : torch.Tensor):
        # We can use F.silu(x) it is the same as swish(x)
        gate_output = self.gate_branch(x)
        return self.w(self.value_branch(x) * (gate_output * self.sigmoid(gate_output)))


# RoFormer: Enhanced Transformer with Rotary Position Embedding https://arxiv.org/pdf/2104.09864 RoPE is applied here
class MultiheadSelfAttention(nn.Module):
    def __init__(self, max_seq_len, h = 32, d_model = 4096, d_k = 4096):
        super().__init__()
        assert d_model % h == 0 , f"Hidden Dimensions {d_model} not divisible by num of heads {h}"
        self.h = h
        self.d_model = d_model
        self.head_dim = d_model // h
        self.linear_q = nn.Linear(in_features=d_model, out_features=d_k)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_k)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_k)
        self.linear_o = nn.Linear(in_features=d_k, out_features=d_model)
        self.rope = RotaryPositionEmbedding(max_seq_len=max_seq_len, head_dim=self.head_dim, base=10000)

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

# GQA:Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints https://arxiv.org/pdf/2305.13245
# 2 main idea to this GQA paper.
# 1. Converting MHA into GQA, which is called uptraining
# 2. Introduces Grouped-query attention which divides query heads into G groups, where each G groups shares 1 single k and v head. GQA-1 == MQA
class GroupQueryAttention(nn.Module):
    def __init__(self, max_seq_len, h, g, d_model):
        super().__init__()
        assert d_model % h == 0, "Embedding Dimension must be divisible by H"
        assert h % g == 0, "Total Number of heads must be divisible by number of group"
        self.h = h # Total no of heads
        self.g = g # Number of groups
        self.head_dim = d_model // self.h # Head Dimension
        self.num_of_heads = self.h // self.g # Number of heads in a group
        self.linear_q = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.linear_k = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.linear_v = nn.Linear(in_features=d_model,out_features=self.head_dim * self.g, bias=False)
        self.linear_o = nn.Linear(in_features=d_model,out_features=d_model, bias=False)
        self.rope = RotaryPositionEmbedding(max_seq_len=max_seq_len, head_dim=self.head_dim, base=10000)

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

# Attention is all you need https://arxiv.org/abs/1706.03762
# Architecture is the same as the decoder used in the original attention paper
# PreNorm Used instead of PostNorm, but residual streams should stay unnormalized
class LlamaDecoder(nn.Module):
    def __init__(self, max_seq_len, h=32, d_model=4096, d_ff=11008):
        super().__init__()
        self.ffn = FFNSwiGLU(d_model=d_model, d_ff=d_ff):
        self.msa = MultiheadSelfAttention(max_seq_len=max_seq_len, h=h, d_model=d_model)
        self.rms_norm = RMSNorm(d_model=d_model):
        self.rms_norm2 = RMSNorm(d_model=d_model): 

    def forward(self, x : torch.Tensor):
        norm_x = self.rms_norm(x)
        mhsa_output, attn_weights = self.msa(norm_x)
        rms2_input = mhsa_output + x
        ffn_input = self.rms_norm2(rms2_input)
        ffn_output = self.ffn(ffn_input)
        output = ffn_output + rms2_input
        return output

class Llama(nn.Module):
    def __init__(self, max_seq_len, n=32, h=32, d_model=4096, d_ff=11008):
        super().__init__()
        self.tkn_embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.decoders = nn.ModuleList([LlamaDecoder(max_seq_len=max_seq_len, h=h, d_model=d_model, d_ff) for _ in range(n)])

    def forward(self, x, torch.Tensor, mask : torch.Tensor = None):
        output = self.tkn_embedding(x)
        for idx,layer in emuerate(self.decoders):
            output = layer(output, mask=mask)

        return output

# LLAMA 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/pdf/2307.09288
# Similar Architecturally to Llama, only difference is GQA instead of MHA
# PreNorm, RoPE, SwiGLU 
# Decoder Only
# Tokenizer still Byte Pair Encoding implemented by SentencePiece
class Llama2Decoder(nn.Module):
    def __init__(self, max_seq_len, h=32, g=6, d_model=4096, d_ff=11008):
        super().__init__()
        self.ffn = FFNSwiGLU(d_model=d_model, d_ff=d_ff):
        self.gqa = GroupQueryAttention(max_seq_len=max_seq_len, h=h, g=g, dmodel=d_model)
        self.rms_norm1 = RMSNorm(d_model=d_model)
        self.rms_norm2 = RMSNorm(d_model=d_model)
    def forward(self, x : torch.Tensor):
        norm_x = self.rms_norm1(x)
        gqa_output, attn_weights = self.gqa(norm_x)
        rms2_input = gqa_output + x
        ffn_input = self.rms_norm2(rms2_input)
        ffn_output = self.ffn(ffn_input)
        output = ffn_output + rms2_input
        return output

class Llama2(nn.Module):
    def __init__(self, max_seq_len, n=32, h=32, g=6,d_model=4096, d_ff=11008):
        super().__init__()
        self.tkn_embedding = nn.Embedding(num_embeddings=vocab_size,embedding
        self.decoders = nn.ModuleList([Llama2Decoder(max_seq_len, h=h, g=g,d_model=d_model, d_ff=d_ff)
 for _ in range(n)])
    
    def forward(self, x, torch.Tensor, mask : torch.Tensor = None):
        output = self.tkn_embedding(x)
        for idx,layer in emuerate(self.decoders):
            output = layer(output, mask=mask)

        return output

@dataclass
class LlamaConfig:
    pass

class Llama2Config:
    pass
