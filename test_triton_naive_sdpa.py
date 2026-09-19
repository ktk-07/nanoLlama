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
               B,
               N,
               row, # S
               col, # S
               inner_dim, # H
               stride_qb,stride_qn,stride_qs,stride_qh,
               stride_kb,stride_kn,stride_kh,stride_ks,
               stride_ob,stride_on,stride_oqs,stride_oks,
               BLOCK_Q:tl.constexpr,
               BLOCK_K:tl.constexpr,
               BLOCK_SIZE:tl.constexpr=1024,
               scaled:tl.constexpr=True
               ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)
        pid2 = tl.program_id(axis=2)

        # Always compute the base memory offset for the selected batch/head
        # 1. Continguous  memory indexing, only works each input and output stored in a row-major format
        # q_offset = pid0 * q_dim[2] * q_dim[3]
        # k_offset = pid0 * k_dim[2] * k_dim[3]
        # o_offset = pid0 * output_dim[2] * output_dim[3]
        # 2. Stride-based Indexing
        # Because we flatten the grid into (B*N,no_of_block_q,no_of_of_block_k)
        b_idx = pid0 // N
        n_idx = pid0 % N
        q_offset = b_idx * stride_qb + n_idx * stride_qn
        k_offset = b_idx * stride_kb + n_idx * stride_kn
        o_offset = b_idx * stride_ob + n_idx * stride_on

        # Get the rows and cols that we want to compute first
        rows = pid1 * BLOCK_Q + tl.arange(0, BLOCK_Q)
        cols = pid2 * BLOCK_K + tl.arange(0, BLOCK_K)

        rows_mask = rows < row
        cols_mask = cols < col

        acc = tl.zeros((BLOCK_Q,BLOCK_K), dtype=tl.float32)

        # Loop through the K
        for i in tl.range(0, inner_dim, BLOCK_SIZE):
            inner_offsets = i + tl.arange(0, BLOCK_SIZE)
            inner_mask = inner_offsets < inner_dim
            Q_mask = rows_mask[:, None] & inner_mask[None, :]
            K_mask = cols_mask[None, :] & inner_mask[:, None]

            Q_IDX_BLOCK = q_offset + rows[:, None] * stride_qs + inner_offsets[None, :] * stride_qh # Assuming rows[:None] means the we unsqueeze to get q x 1, inner_offsets[None:] means we get 1 x inner
            Q_BLOCK = tl.load(q_ptr + Q_IDX_BLOCK, mask=Q_mask, other=0.0)
            K_IDX_BLOCK = k_offset + inner_offsets[:, None] * stride_kh + cols[None, :] * stride_ks # Assuming cols[None:] means the we unsqueeze to get 1 x k, inner_offsets[None:] means we get inner x 1
            K_BLOCK = tl.load(k_ptr + K_IDX_BLOCK, mask=K_mask, other=0.0)

            acc += tl.dot(Q_BLOCK,K_BLOCK)

        O_BlOCK_IDX = o_offset + rows[:, None] * stride_oqs + cols[None, :] * stride_oks
        mask = rows_mask[:, None] & cols_mask[None, :]

        # In S = Q @ K^T / sqrt(d_k)
        if scaled:
            tl.store(output_ptr + O_BLOCK_IDX, acc / tl.sqrt(inner_dim), mask=mask)
        else:
            tl.store(output_ptr + O_BLOCK_IDX, acc, mask=mask)

    def matmul(Q,
               K,
               scaled=True;
               ):
        stride_qb,stride_qn,stride_qs,stride_qh = Q.stride()
        stride_kb,stride_kn,stride_kh,stride_ks = K.stride()
        B_Q, N_Q, S_Q, H_Q = Q.shape
        B_K, N_K, H_K, S_K = K.shape
        assert H_Q == H_K , "Inner dimensions of Tensors not Matching"
        output = torch.empty((B_K, N_K, S_Q, S_K), device=Q.device, dtype=Q.dtype)
        stride_ob,stride_on,stride_oqs,stride_oks = output.stride()

        # Block Sizes does not have to be the same
        BLOCK_Q = 32
        BLOCK_K = 32
        BLOCK_INNER_DIM = 1024
        
        # Launch Grid Replaces Loops that represent independent work; explicit loops remain primarily where there is a dependency/reduction.
        grid = (B_K*N_K, math.ceil(S_Q / BLOCK_Q), math.ceil(S_K, BLOCK_K))
        row = S_Q
        col = S_K
        inner_dim = H_Q
        matmul_kernel[grid](Q, 
                            K, 
                            output, 
                            B_Q, 
                            N_Q, 
                            row, 
                            col, 
                            inner_dim, 
                            stride_qb,stride_qn,stride_qs,stride_qh, 
                            stride_kb,stride_kn,stride_kh,stride_ks,
                            stride_ob,stride_on,stride_oqs,stride_oks,
                            BLOCK_Q,
                            BLOCK_K, 
                            BLOCK_INNER_DIM, 
                            scaled
                            )

        return output

    @triton.jit
    def safe_softmax_kernel(s_ptr,
                       output_ptr,
                       s_dimensions,
                       output_dimensions,
                       BLOCK_SIZE:tl.constexpr=1024
                       ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)
        
        # Get the base offset idx
        row_start = pid0 * output_dimensions[2] * output_dimensions[3]

        # Three Loops
        # Loop 1 to get max
        max_val = float("-inf")
        for i in tl.range(0, output_dimensions[-1], BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < output_dimensions[-1]
            offsets = pid1 * output_dimensions[-1] + cols
            col_idxes = row_start + offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            local_max = tl.max(vals, axis=0)
            max_val = tl.maximum(local_max,max_val)

        # Loop 2 to compute denominator
        denominator = 0.0
        for i in tl.range(0, output_dimensions[-1], BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < output_dimensions[-1]
            offsets = pid1 * output_dimensions[-1] + cols
            col_idxes = row_start + offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            denominator += tl.sum(tl.exp(vals-max_val),axis=0)

        # Loop 3 to compute the actual softmax value and write to output_ptr
        for i in tl.range(0, output_dimensions[-1], BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < output_dimensions[-1]
            offsets = pid1 * output_dimensions[-1] + cols
            col_idxes = row_start + offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            actual = tl.exp(vals - max_val) / denominator
            tl.store(output_ptr + col_idxes, actual, mask=mask)


    # This is the naive version of safe_softmax with no online softmax calculator
    def safe_softmax(S
                     ):
        # In S = Q @ K^T / sqrt(d_k)
        # P = softmax(S)

        # S's shape = B x N x S x H
        # O's shape = B x N x S x H
        B, N, S, H = S.shape
        output = torch.empty_like(s_dimensions, device=S.dtpe, dtype=S.dtype)
        o_dimensions = o.shape
        grid = (B*N, S,) # each program is responsible for 1 row of output
        BLOCK_SIZE = 1024
        
        safe_softmax_kernel[grid](s, output, s_dimensions, output_dimensions, BLOCK_SIZE=BLOCK_SIZE)

        return output
