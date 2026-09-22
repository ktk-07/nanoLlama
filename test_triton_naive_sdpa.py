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
    def matmul_kernel(q_ptr,
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
               scaled:tl.constexpr=False
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
            # On an NVIDIA GPU, Triton was free to use TF32-style tensor-core multiplication for FP32 inputs.
            # If we dont set input_precision="ieee", there would be an error
            acc += tl.dot(Q_BLOCK,K_BLOCK, input_precision="ieee")

        O_BLOCK_IDX = o_offset + rows[:, None] * stride_oqs + cols[None, :] * stride_oks
        mask = rows_mask[:, None] & cols_mask[None, :]

        # In S = Q @ K^T / sqrt(d_k)
        if scaled:
            tl.store(output_ptr + O_BLOCK_IDX, acc / tl.sqrt(inner_dim), mask=mask)
        else:
            tl.store(output_ptr + O_BLOCK_IDX, acc, mask=mask)

    def matmul(Q,
               K,
               scaled=False,
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
        BLOCK_INNER_DIM = 128
        
        # Launch Grid Replaces Loops that represent independent work; explicit loops remain primarily where there is a dependency/reduction.
        grid = (B_K*N_K, triton.cdiv(S_Q, BLOCK_Q), triton.cdiv(S_K, BLOCK_K))
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

    # B x N x S x H
    @triton.jit
    def safe_softmax_kernel_last_dim(s_ptr,
                                     output_ptr,
                                     B,
                                     N,
                                     S,
                                     H,
                                     stride_sb,stride_sn,stride_ss,stride_sh, 
                                     stride_ob,stride_on,stride_os,stride_oh,
                                     BLOCK_SIZE:tl.constexpr=1024
                                     ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)
        
        # Get the base offset idx
        b_idx = pid0 // N
        n_idx = pid0 % N
        s_row_start = b_idx * stride_sb + n_idx * stride_sn
        o_row_start = b_idx * stride_ob + n_idx * stride_on

        # Three Loops
        # Loop 1 to get max
        max_val = float("-inf")
        for i in tl.range(0, H, BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < H
            offsets = pid1 * stride_ss + cols * stride_sh
            col_idxes = s_row_start + offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            local_max = tl.max(vals, axis=0)
            max_val = tl.maximum(local_max,max_val)

        # Loop 2 to compute denominator
        denominator = 0.0
        for i in tl.range(0, H, BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < H
            offsets = pid1 * stride_ss + cols * stride_sh
            col_idxes = s_row_start + offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            denominator += tl.sum(tl.exp(vals-max_val),axis=0)

        # Loop 3 to compute the actual softmax value and write to output_ptr
        for i in tl.range(0, H, BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < H
            offsets = pid1 * stride_ss + cols * stride_sh
            col_idxes = s_row_start + offsets 

            o_offsets = pid1 * stride_os + cols * stride_oh
            o_col_idxes = o_row_start + o_offsets 

            vals = tl.load(s_ptr + col_idxes, mask=mask, other=float("-inf"))
            actual = tl.exp(vals - max_val) / denominator
            tl.store(output_ptr + o_col_idxes, actual, mask=mask)


    # This is the naive version of safe_softmax with no online softmax calculator
    def safe_softmax_last_dim(S
                              ):
        # In S = Q @ K^T / sqrt(d_k)
        # P = softmax(S)

        # S's shape = B x N x S x H
        # O's shape = B x N x S x H
        B, N, s, H = S.shape
        output = torch.empty(S.shape, device=S.device, dtype=S.dtype)
        stride_sb,stride_sn,stride_ss,stride_sh = S.stride()
        stride_ob,stride_on,stride_os,stride_oh = output.stride()
        grid = (B*N, s,) # each program is responsible for 1 row of output
        BLOCK_SIZE = 128

        safe_softmax_kernel_last_dim[grid](S,
                                  output, 
                                  B,
                                  N,
                                  s,
                                  H,
                                  stride_sb,stride_sn,stride_ss,stride_sh, 
                                  stride_ob,stride_on,stride_os,stride_oh,
                                  BLOCK_SIZE=BLOCK_SIZE
                                  )

        return output

    # This is the generalised version of the softmax
    # Note it is fixed-rank but generalised softmax
    # We flatten all other dimensions that is not that dimension along which we want to compute the softmax for
    # Softmax over dimension d
    # One triton program own 1 slice where every coordinate except d is fixed
    @triton.jit
    def safe_softmax_kernel(s_ptr,
                            o_ptr,
                            D0,
                            D1,
                            D2,
                            D3,
                            dim,
                            stride_sb,stride_sn,stride_ss,stride_sh, 
                            stride_ob,stride_on,stride_os,stride_oh,
                            BLOCK_SIZE:tl.constexpr=1024
                            ): 

        pid = tl.program_id(axis=0)
        base_offset = 0
        o_base_offset = 0
        # We are going to loop over it
        reduction_size = 0
        reduction_stride = 0
        o_reduction_stride = 0

        # Since the Dimensions of O and S are the same, we can use the same idxes just use different strides
        if dim == 0:
            reduction_size = D0
            reduction_stride = stride_sb
            o_reduction_stride = stride_ob
            # You wan to compute the stride for s here 
            idx1 = pid // (D2*D3)
            rem = pid % (D2*D3)
            idx2 = rem // D3
            idx3 = rem % D3

            base_offset = idx1 * stride_sn + idx2 * stride_ss + idx3 * stride_sh
            o_base_offset = idx1 * stride_on + idx2 * stride_os + idx3 * stride_oh


        elif dim == 1:
            reduction_size = D1
            reduction_stride = stride_sn
            o_reduction_stride = stride_on
            # You wan to compute the stride for s here
            idx0 = pid // (D2*D3)
            rem = pid % (D2*D3)
            idx2 = rem // D3
            idx3 = rem % D3

            base_offset = idx0 * stride_sb + idx2 * stride_ss + idx3 * stride_sh
            o_base_offset = idx0 * stride_ob + idx2 * stride_os + idx3 * stride_oh

        elif dim == 2:
            reduction_size = D2
            reduction_stride = stride_ss
            o_reduction_stride = stride_os
            # You wan to compute the stride for s here
            idx0 = pid // (D1*D3)
            rem = pid % (D1*D3)
            idx1 = rem // D3
            idx3 = rem % D3

            base_offset = idx0 * stride_sb + idx1 * stride_sn + idx3 * stride_sh
            o_base_offset = idx0 * stride_ob + idx1 * stride_on + idx3 * stride_oh

        else:
            reduction_size = D3
            reduction_stride = stride_sh
            o_reduction_stride = stride_oh
            # You wan to compute the stride for s here
            idx0 = pid // (D1*D2)
            rem = pid % (D1*D2)
            idx1 = rem // D2
            idx2 = rem % D2

            base_offset = idx0 * stride_sb + idx1 * stride_sn + idx2 * stride_ss
            o_base_offset = idx0 * stride_ob + idx1 * stride_on + idx2 * stride_os

        max_val = float("-inf")
        for i in tl.range(0,reduction_size, BLOCK_SIZE):
            reduction_tensor = i + tl.arange(0,BLOCK_SIZE)
            mask = reduction_tensor < reduction_size
            offsets = base_offset + reduction_tensor * reduction_stride

            values = tl.load(s_ptr + offsets, mask=mask, other=float("-inf"))
            local_max = tl.max(values,axis=0)
            max_val = tl.maximum(max_val,local_max)

        denominator = 0.0
        for i in tl.range(0,reduction_size, BLOCK_SIZE):
            reduction_tensor = i + tl.arange(0,BLOCK_SIZE)
            mask = reduction_tensor < reduction_size
            offsets = base_offset + reduction_tensor * reduction_stride

            vals = tl.load(s_ptr + offsets, mask=mask, other=float("-inf"))
            denominator += tl.sum(tl.exp(vals-max_val),axis=0)

        for i in tl.range(0,reduction_size, BLOCK_SIZE):
            reduction_tensor = i + tl.arange(0,BLOCK_SIZE)
            mask = reduction_tensor < reduction_size
            offsets = base_offset + reduction_tensor * reduction_stride
            o_offsets = o_base_offset + reduction_tensor * o_reduction_stride

            vals = tl.load(s_ptr + offsets, mask=mask, other=float("-inf"))
            tl.store(o_ptr + o_offsets, tl.exp(vals-max_val)/denominator, mask=mask)


    def safe_softmax(S,
                     dim=None
                     ):
        output = torch.empty(S.shape, dtype=S.dtype, device=S.device)
        stride_sb,stride_sn,stride_ss,stride_sh = S.stride()
        stride_ob,stride_on,stride_os,stride_oh = output.stride()
        D0, D1, D2, D3 = S.shape
        grid_size = 0

        soft_max_dim = dim
        if soft_max_dim > 4 or soft_max_dim < -4:
            return
        if soft_max_dim < 0:
            soft_max_dim = soft_max_dim + len(S.shape)
        
        grid_size = 1

        for axis, size in enumerate(S.shape):
            if axis != soft_max_dim:
                grid_size *= size

        grid = (grid_size,)
        safe_softmax_kernel[grid](S,
                            output,
                            D0,
                            D1,
                            D2,
                            D3,
                            soft_max_dim,
                            stride_sb,stride_sn,stride_ss,stride_sh, 
                            stride_ob,stride_on,stride_os,stride_oh,
                            BLOCK_SIZE=128
                            )

        return output

    b = 2
    n = 2
    s = 14
    h = 18
    # Testing Softmax
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda": 
        Q = torch.randn(b,n,s,h, device=device, dtype=torch.float32)
        K = torch.randn(b,n,s,h, device=device, dtype=torch.float32)    
        K_tranposed = K.permute(0,1,3,2)
        V = torch.randn(b,n,s,h)    
        O = torch.randn(b,n,s,h)    
        # Testing Matmul
        output1 = matmul(Q,K_tranposed)               
        output2 = Q @ K_tranposed
        print(torch.allclose(output1, output2, atol=1e-5))
        # Testing Softmax
        output3 = safe_softmax_last_dim(output1)
        output4 = torch.softmax(output2,dim=-1)
        output5 = safe_softmax(output1,dim=-1)
        print(torch.allclose(output3, output5, atol=1e-5))
        print("matmul allclose:", torch.allclose(output1, output2, atol=1e-5, rtol=1e-5))

        print("max abs error:",
              (output1 - output2).abs().max().item())

        print("mean abs error:",
              (output1 - output2).abs().mean().item()) 

    else:
        print("Cant test") 
