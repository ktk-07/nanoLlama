import torch
from torch.utils._triton import has_triton
import math

# Brief Recap of Implementing FlashAttention
# 1. Online normalizer calculation for softmax
# Speeds up safe softmax by reducing its memory accesses/passes
# Keeps 2 statistics
#   - running max
#   - running normalization denominator l
# Key Idea : Softmax statistics can be updated without seeing the entire input at once
#
# 2. Self-Attention does not need o(n^2)
# Extend online softmax to online attention
# Implements attention by fusing online softmax with the matmul operation with V so that attention does not materialise (S and P matrix which is O(n^2 in memory) )
# Keeps 3 statistics
#   - running max (you have to compute local max locally and update it yourself)
#   - running deminator
# 3. Flash-Attention
# Make use of the SRAM cache which is on Chip, computation is still O(N^2) but reduces alot of HBM accesses as compared to 2.
# Make use of block tiling technique

# Phase 1, Just make it work on fixed dimensions first
# 1. Implement Safe Softmax Kernel
# 2. Implement Online Softmax Kernel
# 3. Implement Scaled Dot Product Attention Kernel
# 4. Implement Memory Efficient Attention Kernel (Online Softmax Fused with Scaled Dot Product Attention)
# 5. Implement Flash Attn Forward Kernel
# 6. Implement Flash Attn Backward Kernel

# Phase 2, Just make the different functions work on arbitary dimension
if not has_triton:
    print("Skipping because triton is not installed in this machine")
else:
    import triton
    import triton.language as tl

    # Fixed Input Version First
    # Original B x S X H
    # Now B x N x S x H

    # Softmax is a unary operation
    @triton.jit
    def safe_softmax_triton_kernel(x_ptr,
                                   output_ptr,
                                   dimensions,
                                   reduction_axis,
                                   n_elements,
                                   BLOCK_SIZE
                                   ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)

        row_idx = pid0 * dimensions[2] + pid1
        row_start = row_idx * dimensions[3]

        # 3 Loops
        # 1 Loop for the global_max in that dimension
        # 1 Loop for denominator
        # 1 Loop for the computation
        
        # 1 program loads 1 row
        # This codeblock here is sequential (similar to python' range) but using the tl.range lets trition
        # → Triton runtime/compiler IR loop
        # → good when loop bounds depend on runtime values
        # → exposes loop optimization controls to Triton 

        global_max = float("-inf") # unless there is another way to represent it
        for i in tl.range(0, dimensions[reduction_axis], BLOCK_SIZE): 
            cols = i + tl.arange(0, BLOCK_SIZE)
            # Create a mask to guard memory operations against out-of-bounds accesses.
            mask = cols < dimensions[reduction_axis]
            offsets = row_start + cols
            # Load x from DRAM, masking out any extra elements in case the input is not a multiple of the block size.
            cur_block_vals = tl.load(x_ptr + offsets,mask=mask, other=-float('inf'))
            local_max = tl.max(cur_block_vals, axis=0)
            global_max = tl.maximum(global_max,local_max)

        denominator = 0.0
        for i in tl.range(0, dimensions[reduction_axis], BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < dimensions[reduction_axis]
            offsets = row_start + cols
            # Load x from DRAM, masking out any extra elements in case the input is not a multiple of the block size.
            cur_block_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
            local_sum = tl.sum(tl.exp(cur_block_vals - global_max), axis=0)
            denominator += local_sum

        # We will run the loop 1 more time to compute the actual softmaxed value
        for i in tl.range(0, dimensions[reduction_axis], BLOCK_SIZE):
            cols = i + tl.arange(0,BLOCK_SIZE)
            mask = cols < dimensions[reduction_axis]
            offsets = row_start + cols
            cur_block_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

            # Write Output to DRAM
            tl.store(output_ptr + offsets, tl.exp(cur_block_vals - global_max) / denominator, mask=mask)

    def safe_softmax_triton(x, dim=-1):
        # Similar to CUDA
        # 1. We need to preallocate the outputs
        #   - need to extract out the dim that we are reducing over
        #   - lets just assume we always reduce over the last dimension first
        output = torch.empty_like(x.shape) # Let just assume we alway reduce over the last dimension first

        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        # 2. Similar to how in cuda/c we define dim3 threadDim, blockDim, gridDim
        # We basically need to define the launch grid, it is at most 3D
        # Mapping the program instance to tensor
        B,N,S,H = x.shape
        #grid = lambda x : ()
        grid = (B*N,S,) 
        dimensions = x.shape
        reduction_axis = 3

        # 3. Each program instance will run 1 block of N elements
        safe_softmax_triton_kernel[grid](x,output,dimensions,reduction_axis,n_elements,BLOCK_SIZE=H)
        return output

    # Online normalizer calculation for softmax: https://arxiv.org/pdf/1805.02867
    @triton.jit
    def online_softmax_triton_kernel(x_ptr,
                                   output_ptr,
                                   dimensions,
                                   reduction_axis,
                                   n_elements,
                                   BLOCK_SIZE
                                   ):
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)

        row_idx = pid0 * dimensions[2] + pid1
        row_start = row_idx * dimensions[3]

        # First loop to keep track of Running Statistics
        running_max = float("-inf")
        running_denominator = 0.0

        for i in tl.range(0, dimensions[reduction_axis], BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < dimensions[reduction_axis]
            offsets = row_start + cols
            cur_block_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))

            prev_max = running_max
            local_max = tl.max(cur_block_vals, axis=0)
            # Update running max
            running_max = tl.maximum(local_max,running_max)
            running_denominator *= tl.exp(prev_max - running_max)
            # cur_val - running max
            running_denominator += tl.sum(tl.exp(cur_block_vals - running_max), axis=0)

        # We will run the loop 1 more time to compute the actual softmaxed value
        for i in tl.range(0, dimensions[reduction_axis], BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < dimensions[reduction_axis]
            offsets = row_start + cols
            cur_block_vals = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
            # Write Output to DRAM
            tl.store(output_ptr + offsets, tl.exp(cur_block_vals - running_max) / running_denominator, mask=mask)

    def online_softmax_triton(x, dim=-1):

        output = torch.empty_like(x.shape)

        n_elements = output.numel()
        B,N,S,H = x.shape
        grid = (B, N*S,) 
        dimensions = x.shape
        reduction_axis = 3

        online_softmax_triton_kernel[grid](x,output,dimensions,reduction_axis,n_elements,BLOCK_SIZE=H)
        return output


    # Attention is all you need : https://arxiv.org/pdf/1706.03762
    @triton.jit
    def scaled_dot_product_attn_kernel(
                                      q_ptr,
                                      k_ptr,
                                      v_ptr,
                                      s_ptr,
                                      p_ptr,
                                      o_ptr,
                                      input_dimensions,
                                      output_dimensions,
                                      BLOCK_SIZE=1024
                                      ):

        # Matrix Multiplication S = Q @ K^T
        # How do we want to speed up the matrix multiplication?
        # Load a block of row of Q and loadd a block of row of K 
        # Then load through the 

        # Need to Index K in a Column-Major Manner
        # Optimisation we can make to Matmul
        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)
        pid2 = tl.program_id(axis=2)

        # B x N x S x H
        # B*N x S x H
        # Grid (B*H, S, )

        row_idx_q = pid0 * input_dimensions[2] + pid1
        row_start_q = row_idx_q * input_dimensions[3]

        row_idx_k = pid0 * input_dimensions[2] + pid2
        row_start_k = row_idx_q * input_dimensions[3]

        # So 1 program has 1 row of Q and 1 row of K (K tranpose shape is B x N x H x S, then we want the column of the tranpose which is jus thw row of K)
        # basically think of it like row vector and row vector dot product
        # Thus i think we dont need these 2 lines here where by we do column-major indexing, but if we were to do column based indexing would this be correct?
        #col_idx = pid0 * input_dimensions[2]
        #col_start =

        accumulator = 0.0     
        for i in tl.range(0, input_dimensions[3], BLOCK_SIZE):
            # We need to accumulate then store
            idxes_along_k = i + tl.arange(0, BLOCK_SIZE)
            masks = idxes_along_k < input_dimensions[3]
            offsets = row_start_q + idxes_along_k
            offsets_k = row_start_k + idxes_along_k
            q = tl.load(q_ptr+offsets,mask=mask, other=0.0)
            k = tl.load(k_ptr+offsets_k,mask=mask, other=0.0)
            accumulator += tl.sum(q*k,axis=0)

        # Feels like im not indexing correctly too
        # Rescaled S / sqrt(H)
        #tl.store(s_ptr + row_idx, accumulator / tl.sqrt(input_dimensions[3], total_sum) )

        # Apply Mask 

        # Softmax to get P

        # O = P @ V


        pass

        # It should be 3 separate kernels
        # 1. safe_softmax (No point for it to be the optimised softmax from the other 2 prerequisisite papers)
        # 2. matmul (It can and should be the most optimsed) Reused for QK^T and 0 = PV
        # Kernel 1: S = Q @ K^T / sqrt(H)
        # Kernel 2: P = softmax(S + mask)   # or apply mask before softmax
        # Kernel 3: O = P @ V
    def scaled_dot_product_attn(Q,
                                K,
                                V,
                                mask
                                ):
        # Defining the outputs as well as intermediate tensors required
        B, N, S, H = Q.shape
        S = torch.empty_like((B,N,S,S))
        P = torch.empty_like((B,N,S,S))
        O = torch.empty_like(Q.shape)
        grid = (B*N, S, H) # Grid should be actual work/output tiles your kernel is responsible 



        return scaled_dot_product_attn[grid](Q,K,V,S,P,O, Q.shape, O.shape, BLOCK_SIZE=1024)


    # SELF-ATTENTION DOES NOT NEED O(n^2) MEMORY: https://arxiv.org/pdf/2112.05682
    # I will implement this last, after flash attention forward and backward
    @triton.jit
    def memory_efficient_attn_kernel():
        # 1. We dont want to materialise the S and the P 
        # 2. 
        pass

    def memory_efficient_attn(Q,K,V,mask):
        # grid = ()
        # return memory_efficient_attn_kernel[grid](Q,K,V)
        pass

    # Compare Speed of softmax in pytorch and triton
    # We will do it without accomodating the mask first
    # 1 triton program owns one block of query tokens for one batch_head pair
    # Each program is responsible for BLOCK_Q query tokens and their complete attention output over all keys
    @triton.jit
    def dense_flash_attn_kernel(q_ptr,
                          k_ptr, # this is the k_tranposed, not too sure if thats how they are doing it
                          v_ptr,
                          output_ptr,
                          softmax_scale,
                          N,
                          row,
                          col,
                          inner_dim,
                          stride_qb, stride_qn, stride_qs, stride_qh,
                          stride_kb, stride_kn, stride_kh, stride_ks,
                          stride_vb, stride_vn, stride_vs, stride_vh,
                          stride_ob, stride_on, stride_os, stride_oh,

                          BLOCK_Q: tl.constexpr,
                          BLOCK_KV: tl.constexpr,
                          BLOCK_INNER_DIM: tl.constexpr, 
                          #TILE_Q: tl.constexpr, # T_r not used as it is handled by launch grid
                          #TILE_KV: tl.constexpr, # T_c not used as it is handled by K/V loop
                          ):

        pid0 = tl.program_id(axis=0)
        pid1 = tl.program_id(axis=1)

        # Need to compute the base offset first
        # Because we flatten B * N we need to extract them out
        b_idx = pid0 // N
        n_idx = pid0 % N

        q_base_offset = b_idx * stride_qb +  n_idx * stride_qn
        k_base_offset = b_idx * stride_kb +  n_idx * stride_kn
        v_base_offset = b_idx * stride_vb +  n_idx * stride_vn
        o_base_offset = b_idx * stride_ob +  n_idx * stride_on

        # We can compute the row outside cause the inner loop is only for K/V
        rows = pid1 * BLOCK_Q + tl.arange(0, BLOCK_Q)
        rows_block_mask = rows < row

        # Inner Loop K/V Loop does not use this
        inner = tl.arange(0,BLOCK_INNER_DIM)
        inner_block_mask = inner < inner_dim

        row_mask = rows_block_mask[:,None] & inner_block_mask[None,:]
        row_idxes = q_base_offset + rows[:,None] * stride_qs + inner[None,:] * stride_qh
        q =  tl.load( q_ptr + row_idxes, mask=row_mask, other=0.0)

        # Accumulator should just be BLOCK_Q x HEAD_DIM
        final_acc = tl.zeros((BLOCK_Q, BLOCK_INNER_DIM), dtype=tl.float32)
        global_max = tl.full((BLOCK_Q, ), float("-inf"), dtype=tl.float32)
        global_denominator = tl.zeros((BLOCK_Q, ),dtype=tl.float32)
        # We want to do the kernel fusion for the entire inner_dim reight?
        # Both matmuls happen inside the same K/V-loop iteration.
        for j in tl.range(0, col, BLOCK_KV):
            cols = j + tl.arange(0,BLOCK_KV)
            cols_block_mask = cols < col

            # Accumulator should it be BLOCK_Q by BLOCK_KV or just BLOCK_Q x 1
            # Reading the FlashAttention paper its say B_r x B_c, so this is correct
            matmul_acc = tl.zeros((BLOCK_Q,BLOCK_KV), dtype=tl.float32)
            # Matmul one BLOCK_Q of QK^T first
            # Im not going to optimise it first with like group ordering, i think we would have to somehow change the way we traverse a loop?
            # We want to do the kernel fusion for the entire inner_dim right?

            # Matmul + Online Softmax in this loop 
            # Online Softmax, need update max, deminator, unormalised accumulator
            # Removed Inner loop due to that fact that BLOCK_INNER_DIM > inner_dim (always for flash attention)

            col_mask =  cols_block_mask[None,:] & inner_block_mask[:,None]
            col_idxes = k_base_offset + cols[None,:] * stride_ks + inner[:,None] * stride_kh

            k_transposed = tl.load( k_ptr + col_idxes, mask=col_mask, other=0.0)
            matmul_acc += tl.dot(q,k_transposed, input_precision="ieee")

            # Attention Scaling by sqrt of dimensions
            # softmax_scale computed outside kernel, preventing runtime type problems
            matmul_acc = matmul_acc * softmax_scale
            # We need this because final K/V block may contain fewer than BLOCK_KV valid keys.
            matmul_acc = tl.where(cols_block_mask[None, :],matmul_acc,-float("inf"))

            # Online Rescaling 3 Running Statistic/States to Keep, technically 6 cause 3 for local then 3 for global meaning the entire 1 x embedding_dim row
            # I think 3 Running Statisic is fine because block_inner_dim > inner_dim so there is no inner loop for block_inner_dim
            # 1. runnning max
            # 2. running denominator
            # 3. running unnormalized weighted values

            # Getting the local max of the row, BLOCK_Q X 1
            local_max = tl.max(matmul_acc, axis=-1) 

            # Rescale old statistic, denominator and accumulator with (new_max and global_max which we have not updated)
            new_max = tl.maximum(global_max, local_max)
            rescaling_factor = tl.exp(global_max - new_max)
            global_denominator *= rescaling_factor
            final_acc *= rescaling_factor[:, None]

            # Update global_max
            global_max = new_max

            # Accumulate the value into the global denominator
            # p = tl.exp(matmul_acc - global_max[:, None]), i dont really want to initallise this im afraid it might use registers
            matmul_acc = tl.exp(matmul_acc - global_max[:, None])
            global_denominator += tl.sum(matmul_acc, axis=-1)

            # Compute P @ V_j [BLOCK_Q,BLOCK_INNER_DIM]
            v_inner = tl.arange(0, BLOCK_INNER_DIM)
            v_inner_mask =  v_inner < inner_dim
            v_mask = cols_block_mask[:, None] & v_inner_mask[None, :]

            v_idxes = v_base_offset + cols[:, None] * stride_vs + v_inner[None, :] * stride_vh

            v = tl.load( v_ptr + v_idxes, mask=v_mask, other=0.0) 
            # This is the local accumulator
            final_acc += tl.dot(matmul_acc.to(v.dtype), v, input_precision="ieee")

        # We want to write to O here outside the K/V loop
        # We need to divide unnormalised A by the global deminator
        # We now have the global max
        # We now have the final weighted unnormalised accumulated values
        o_mask = rows_block_mask[:, None] & inner_block_mask[None, :] # Not needed
        o_idxes = o_base_offset + rows[:, None] * stride_os + inner[None, :] * stride_oh
        output = final_acc / global_denominator[:, None]
        tl.store(output_ptr + o_idxes, output, mask=o_mask)

    # Compare Speed of softmax in pytorch and triton
    def dense_flash_attn(Q, K, V):

        # In the paper S = N (sequence length) and H = d (head dimension)
        # Q: [B, N, S_Q, H]
        # K: [B, N, S_KV, H]
        # V: [B, N, S_KV, H]

        B_Q, N_Q, S_Q, H_Q = Q.shape
        B_K, N_K, S_K, H_K = K.shape
        B_V, N_V, S_V, H_V = V.shape

        assert B_Q == B_K == B_V
        assert N_Q == N_K == N_V
        assert H_Q == H_K == H_V
        assert S_K == S_V

        # We want the transposed view of K
        # Original K:   [B, N, S_K, H]
        # Transposed K: [B, N, H, S_K]
        # Transpose returns a view, so we must use its actual strides.
        K_transposed = K.transpose(-2, -1)

        # Explicitly extract strides for educational purposes.
        # Memory address = base pointer + sum(index_i * stride_i)
        #
        # Q[b,n,s,h] -> b*stride_qb + n*stride_qn + s*stride_qs + h*stride_qh
        stride_qb, stride_qn, stride_qs, stride_qh = Q.stride()

        # K_transposed[b,n,h,s] -> b*stride_kb + n*stride_kn + h*stride_kh + s*stride_ks
        stride_kb, stride_kn, stride_kh, stride_ks = K_transposed.stride()

        # V[b,n,s,h] -> b*stride_vb + n*stride_vn + s*stride_vs + h*stride_vh
        stride_vb, stride_vn, stride_vs, stride_vh = V.stride()

        output = torch.empty(
            (B_Q, N_Q, S_Q, H_Q),
            dtype=Q.dtype,
            device=Q.device
        )

        # Output has the same logical shape as Q: [B, N, S_Q, H]
        stride_ob, stride_on, stride_os, stride_oh = output.stride()

        # Since each program is responsible for BLOCK_Q query tokens
        # and their complete attention output over all keys. 
        # Original FlashAttention paper:
        # M = available SRAM capacity (number of elements)
        # B_r = number of query rows in one Q block
        # B_c = number of K/V rows in one K/V block
        #
        # Paper uses:
        # B_c = ceil(M / (4*d))
        # B_r = min(ceil(M / (4*d)), d)
        #
        # For now, choose fixed block sizes for correctness testing.
        # Later we can tune them based on SRAM usage and performance.
        BLOCK_Q = 32       # B_r
        BLOCK_KV = 32      # B_c

        # Head dimension is processed in full.
        # Pad to a power of 2 because tl.arange requires it.
        BLOCK_INNER_DIM = max(16, triton.next_power_of_2(H_Q))

        # Compute softmax scale 
        softmax_scale = 1.0 / math.sqrt(H_Q)

        # Tile is the number of blocks required to cover the sequence.
        TILE_Q = triton.cdiv(S_Q, BLOCK_Q)   # T_r
        TILE_KV = triton.cdiv(S_K, BLOCK_KV)  # T_c

        # Each program owns one Q block for one batch/head pair.
        # TILE_Q is handled by the launch grid.
        # TILE_KV is handled by the K/V loop inside the kernel.
        grid = (B_Q * N_Q, TILE_Q)

        dense_flash_attn_kernel[grid](
            Q,
            K_transposed,
            V,
            output,
            softmax_scale,

            N_Q,
            S_Q,       # row = Q sequence length
            S_K,       # col = K/V sequence length
            H_Q,       # inner_dim = head dimension

            stride_qb, stride_qn, stride_qs, stride_qh,
            stride_kb, stride_kn, stride_kh, stride_ks,
            stride_vb, stride_vn, stride_vs, stride_vh,
            stride_ob, stride_on, stride_os, stride_oh,

            BLOCK_Q=BLOCK_Q,
            BLOCK_KV=BLOCK_KV,
            BLOCK_INNER_DIM=BLOCK_INNER_DIM,
        )

        return output

    def flash_attention():
        pass

    def flash_attention():
        pass

    def flash_attention_backward_kernel():
        pass

    def flash_attention_backward():
        pass


    # Testing FlashAttention Forward
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.manual_seed(42)

        dtype = torch.float32

        # (B, N, S_Q, S_KV, H)
        # Test different batch sizes, heads, sequence lengths,
        # and head dimensions.
        test_cases = [
            (1, 1, 32, 32, 32),
            (1, 1, 64, 64, 64),
            (1, 1, 128, 128, 64),
            (1, 1, 130, 130, 64),   # Non-divisible sequence length
            (1, 1, 257, 257, 64),   # Partial Q and K/V blocks
            (2, 4, 128, 128, 64),   # Multiple batches and heads
            (2, 8, 256, 256, 64),
            (1, 2, 128, 192, 64),   # Different Q and K/V lengths
            (1, 2, 65, 97, 32),
            (1, 2, 129, 129, 80),   # Head dimension padding
        ]

        print("FlashAttention Forward Correctness Tests")
        print("-" * 100)

        for B, N, S_Q, S_KV, H in test_cases:

            Q = torch.randn(
                B, N, S_Q, H, device=device, dtype=dtype
            )
            K = torch.randn(
                B, N, S_KV, H, device=device, dtype=dtype
            )
            V = torch.randn(
                B, N, S_KV, H, device=device, dtype=dtype
            )

            # PyTorch reference implementation
            # Non-causal attention, no dropout
            reference = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=0.0,
                is_causal=False
            )

            # Our Triton FlashAttention forward for the dense forward
            actual = dense_flash_attn(Q, K, V)

            # Compare numerical correctness
            max_error = (actual - reference).abs().max().item()
            mean_error = (actual - reference).abs().mean().item()

            passed = torch.allclose(
                actual,
                reference,
                atol=1e-4,
                rtol=1e-4
            )

            print(
                f"B={B}, N={N}, "
                f"S_Q={S_Q}, S_KV={S_KV}, H={H} | "
                f"{'PASS' if passed else 'FAIL'} | "
                f"Max Error: {max_error:.6e} | "
                f"Mean Error: {mean_error:.6e}"
            )

            assert passed, (
                f"FlashAttention failed for "
                f"B={B}, N={N}, S_Q={S_Q}, S_KV={S_KV}, H={H}"
            )

        print("-" * 100)
        print("All FlashAttention forward correctness tests passed!")

    else:
        print("CUDA is not available. Skipping Triton FlashAttention tests.")