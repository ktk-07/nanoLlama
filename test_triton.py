import torch
from torch.utils._triton import has_triton

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
# 1. Implement Safe Softmax 
# 2. Implement Online Softmax
# 3. Implement Online Attention (Online Softmax Fused with Online Attention) so that attention does not materialise
# 4. Implement Flash Attn Forward
# 5. Implement Flash Attn Backward

# Phase 2, Just make the different functions work on arbitary dimension
if not has_triton:
    print("Skipping because triton is not installed in this machine")
else:
    import triton
    import triton.language as tl

    # Fixed Input Version
    # B x S X H, I know in scaled dot product attention its B x N x S x H

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

        row_idx = pid0 * dimensions[1] + pid1
        row_start = row_idx * dimensions[2]

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
            local_sum = tl_sum(tl.exp(cur_block_vals - global_max), axis=0)
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
        output = torch.empty_like(x.shape[x:-1) # Let just assume we alway reduce over the last dimension first

        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        # 2. Similar to how in cuda/c we define dim3 threadDim, blockDim, gridDim
        # We basically need to define the launch grid, it is at most 3D
        # Mapping the program instance to tensor
        B,S,H = x.shape
        #grid = lambda x : ()
        grid = (B,S,) 
        dimensions = x.shape
        reduction_axis = 2

        # 3. Each program instance will run 1 block of N elements
        safe_softmax_triton_kernel[grid](x,output,dimensions,reduction_axis,n_elements,BLOCK_SIZE=H)
        return output

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

        row_idx = pid0 * dimensions[1] + pid1
        row_start = row_idx * dimensions[2]

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

        output = torch.empty_like(x.shape[x:-1) # Let just assume we alway reduce over the last dimension first

        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        B,S,H = x.shape
        #grid = lambda x : ()
        grid = (B,S,) 
        dimensions = x.shape
        reduction_axis = 2

        online_softmax_triton_kernel[grid](x,output,dimensions,reduction_axis,n_elements,BLOCK_SIZE=H)
        return output

    # Compare Speed of softmax in pytorch and triton
    def scaled_dot_product_attn_flash():
        pass


    inputs = torch. 



