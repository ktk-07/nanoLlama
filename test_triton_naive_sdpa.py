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
    # When writing kernels in triton, the shape-matching idea is a very useful Triton habit
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

        # Always compute the base memory offset for the selected batch/head
        q_offset = pid0 * q_dim[2] * q_dim[3]
        k_offset = pid0 * k_dim[2] * k_dim[3]
        o_offset = pid0 * output_dim[2] * output_dim[3]

        # Get the rows and cols that we want to compute first
        rows = pid1 * BLOCK_Q + tl.arange(0, BLOCK_Q)
        cols = pid2 * BLOCK_K + tl.arange(0, BLOCK_K)

        rows_mask = rows < q_dim[-2]
        cols_mask = cols < k_dim[-1]

        acc = tl.zeros((BLOCK_Q,BLOCK_K), dtype=tl.float32)

        # Loop through the K
        for i in tl.range(0, INNER_DIM, BLOCK_SIZE):
            inner_offsets = i + tl.arange(BLOCK_SIZE)
            inner_mask = inner_offsets < INNER_DIM
            Q_mask = rows_mask[:, None] & inner_mask[None, :]
            K_mask = cols_mask[None, :] & inner_mask[:, None]

            Q_IDX_BLOCK = q_offset + rows[:, None] * q_dim[-1] + inner_offsets[None, :] # im assuming rows[:None] means the we unsqueeze to get q x 1, inner_offsets[None:] means we get 1 x inner
            Q_BLOCK = tl.load(q_ptr + Q_IDX_BLOCK, mask=Q_mask, other=0.0)
            K_IDX_BLOCK = k_offset + cols[None, :] + k_dim[-1] * inner_offsets[:, None] # Im assuming cols[None:] means the we unsqueeze to get 1 x k, inner_offsets[None:] means we get inner x 1
            K_BLOCK = tl.load(k_ptr + K_IDX_BLOCK, mask=K_mask, other=0.0)

            acc += tl.dot(Q_BLOCK,K_BLOCK)

        O_BlOCK_IDX = o_offset + rows[:, None] * output_dim[-1] + cols[None, :] 
        mask = rows_mask[:, None] & cols_mask[None, :]
        tl.store(output_ptr + O_BLOCK_IDX, acc, mask=mask)

    def matmul(Q,
               K
               ):
        B_Q, N_Q, S_Q, H_Q = Q.shape
        B_K, N_K, H_K, S_K = K.shape
        assert H_Q == H_K , "Inner dimensions of Tensors not Matching"
        output = torch.empty_like(B_K, N_K, S_Q, S_K, device=Q.device, dtype=Q.dtype)

        # Block Sizes does not have to be the same
        BLOCK_Q = 32
        BLOCK_K = 32
        BLOCK_INNER_DIM = 1024
        
        # Launch Grid Replaces Loops that represent independent work; explicit loops remain primarily where there is a dependency/reduction.
        grid = (B_K*N_K, math.ceil(S_Q / BLOCK_Q), math.ceil(S_K, BLOCK_K))

        no_of_elements = O.numel()

        matmul_kernel[grid](Q, K, output, no_of_elements, BLOCK_Q, BLOCK_K, BLOCK_INNER_DIM)

        return output

    @triton.jit
    def softmax_kernel(s_ptr,
                       output_ptr,
                       s_dimensions,
                       ):
        pass

    def softmax(s
                ):
        output = None
        return output
