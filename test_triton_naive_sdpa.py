import torch
from torch.utils._triton import has_triton

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    # Implementing the naive version of scaled dot product attention
    # We launch 3 separate kernels
    # Q, K, V 's shape is B x N x S x H
    # S = Q @ K^T
    # P = softmax (S/sqrt(H))
    # O = P @ V

    # Implementing Matmul

    @triton.jit
    def matmul(q_ptr,
               k_ptr,
               output_ptr,
               q_dim,
               k_dim,
               output_dim,
               no_of_elements,
               BLOCK_Q,
               BLOCK_K,
               BLOCK_SIZE=1024
               ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)
        pid2 = tl.program_id(axis=2)

        INNER_DIM = q_dim[-1]

        # this matmul is assuming K is already transposed, im not too sure what the fused version should do? im assuming they don know expect K to be fused??

    def matmul_kernel(Q,
                      K
                      ):
        B_Q, N_Q, S_Q, H_Q = Q.shape
        B_K, N_K, H_K, S_K = K.shape
        assert H_Q == H_K , "Inner dimensions of Tensors not Matching"
        output = torch.empty_like(B_K, N_K, S_Q, S_K)

        # Block Sizes does not have to be the same
        BLOCK_Q = 32
        BLOCK_K = 32
        BLOCK_INNER_DIM = 1024
        
        # a launch grid replaces loops that represent independent work; explicit loops remain primarily where there is a dependency/reduction.
        grid = (B_K*N_K, math.ceil(S_Q / BLOCK_Q), math.ceil(H_K, BLOCK_K))

        no_of_elements = O.numel()

        matmul[grid](Q, K, output, no_of_elements, BLOCK_Q, BLOCK_K, BLOCK_INNER_DIM)

        return output

