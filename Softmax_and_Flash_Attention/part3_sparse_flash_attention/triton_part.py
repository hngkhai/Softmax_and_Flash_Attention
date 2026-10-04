import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Helper function
def pow2_floor(x: int) -> int:
    x = max(x, 16)
    return 1 << (x.bit_length() - 1)

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr, sparse_mask_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Dimensions
    pid_m = tl.program_id(0)
    num_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    
    # Offsets for Q
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d_model)
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :]
    
    # Load Q
    # Masking for Q is needed if seq_len is not a multiple of BLOCK_M
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0).to(tl.float32)

    # Initialize accumulation variables
    o = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    m = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    
    scale_factor = 1.0 / (d_model ** 0.5)
    
    # Iterate over K, V blocks (looping over N)
    # We maintain a block index to check against the sparse_mask_ptr
    block_n_idx = 0
    
    for start_n in range(0, seq_len, BLOCK_N):
        # Check the sparse mask before doing any heavy computation
        # The mask is flat: [num_blocks_m, num_blocks_n]
        mask_offset = pid_m * num_blocks_n + block_n_idx
        is_active = tl.load(sparse_mask_ptr + mask_offset).to(tl.int1)
        
        # Only compute if the mask allows it
        if is_active:
            offs_n = start_n + tl.arange(0, BLOCK_N)
            
            # Pointers for K and V
            k_ptrs = k_ptr + offs_n[:, None] * stride_km + offs_d[None, :]
            v_ptrs = v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :]
            
            # Load K and V
            # Masking for K/V is needed if seq_len is not a multiple of BLOCK_N
            col_mask = (offs_n < seq_len)
            k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0).to(tl.float32)
            v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.0).to(tl.float32)
            
            # --- Attention Core ---
            s = tl.dot(q, tl.trans(k)) * scale_factor
            
            # Apply masking for out-of-bounds keys
            s = tl.where(col_mask[None, :], s, -float('inf'))

            # Compute stats
            m_ij = tl.max(s, axis=1)
            p_ij = tl.exp(s - m_ij[:, None])
            l_ij = tl.sum(p_ij, axis=1)

            # Online Softmax Update
            m_new = tl.maximum(m, m_ij)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(m_ij - m_new)

            l = alpha * l + beta * l_ij
            # Update output accumulator
            o = alpha[:, None] * o + beta[:, None] * tl.dot(p_ij, v)
            
            # Update max score
            m = m_new

        # Increment block index for next iteration
        block_n_idx += 1

    # Final normalization
    inv_l = tl.where(l > 0, 1.0 / l, 0.0)
    o = o * inv_l[:, None]

    # Store output
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, o, mask=(offs_m[:, None] < seq_len))

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def get_block_MN(d_model):
    # Logic adapted from using global properties 
    if d_model <= 64:
        BLOCK_M = 64
    elif d_model <= 128:
        BLOCK_M = 64
    else:
        BLOCK_M = 32

    # Calculate optimal BLOCK_N based on shared memory
    smem_limit = SIZE_SMEM
    safety = 0.9
    num_stages = 2 # Default for older Triton versions or explicit pipelines
    bytes_per = 4  # float32
    
    max_bn = int((smem_limit * safety) // (2 * d_model * bytes_per * num_stages))
    max_bn = max(16, min(128, max_bn))
    BLOCK_N = pow2_floor(max_bn)

    return BLOCK_M, BLOCK_N

def call_flash_attention_v1_sparse(q, k, v, mask_ptr):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert q.dim() == 2, "Only support 2D input: (seq_len, d_model)"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)
    
    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    
    # Grid definition
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    
    # Heuristics for warps/stages 
    num_warps = 4 if d_model <= 128 else 8
    
    flash_attention_v1[grid](
        q, k, v, o, mask_ptr,
        seq_len, d_model,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=1 
    )
        
    return o

def pytorch_attention(q, k, v):
        d_model = q.shape[-1]
        scale_factor = 1.0 / (d_model**0.5)
        scores = q @ k.T * scale_factor
        softmax = torch.softmax(scores, dim=1)
        attention = softmax @ v
        return attention
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_len'],
        x_vals=[ 2 ** i for i in range(7, 13)],
        line_arg='d_model',
        line_vals=[64, 128, 256],
        line_names=['d=64', 'd=128', 'd=256'],
        styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
        ylabel="Latency (ms)",
        plot_name="flash-attention-performance",
        args={'provider': 'flash_attention'},
    )
)
def benchmark(seq_len, d_model, provider):
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    mask = generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N)
    
    if provider == 'flash_attention':
        ms = triton.testing.do_bench(lambda: call_flash_attention_v1_sparse(q, k, v, mask))
    return ms

def unit_test(seq_len, d_model):
    torch.manual_seed(0)
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    BLOCK_M, BLOCK_N = get_block_MN(d_model)
    mask = generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N)
    
    o_triton = call_flash_attention_v1_sparse(q, k, v, mask)
    o_native = native_block_sparse_attention(q, k, v, mask, BLOCK_M, BLOCK_N)

    assert torch.allclose(o_triton, o_native, atol=2e-3, rtol=2e-3), (o_triton, o_native)
    print(f"Attention output correct for seq_len={seq_len}, d_model={d_model}!")

def generate_block_sparse_mask(seq_len, BLOCK_M, BLOCK_N):
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    mask = torch.zeros((num_blocks_m, num_blocks_n), dtype=torch.int8, device='cuda')
    
    for i in range(num_blocks_m):
        if i < num_blocks_n:
            mask[i, i] = 1
        for k in range(int(torch.log2(torch.tensor(num_blocks_n))) + 1):
            j = i ^ (1 << k)  
            if j < num_blocks_n:
                mask[i, j] = 1
                
    return mask.view(-1) 

def native_block_sparse_attention(q, k, v, sparse_mask_flat, BLOCK_M, BLOCK_N):
    seq_len, d_model = q.shape
    scale = 1.0 / (d_model ** 0.5)

    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    assert sparse_mask_flat.numel() == num_blocks_m * num_blocks_n, f"sparse_mask_flat legnth wrong!"

    block_mask = sparse_mask_flat.view(num_blocks_m, num_blocks_n)
    
    m_idx = torch.arange(seq_len, device=q.device) // BLOCK_M  
    n_idx = torch.arange(seq_len, device=q.device) // BLOCK_N  
    m_idx_expand = m_idx.unsqueeze(1).expand(-1, seq_len)
    n_idx_expand = n_idx.unsqueeze(0).expand(seq_len, -1)
    expanded_mask = block_mask[m_idx_expand, n_idx_expand]
    
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale 
    scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
    mask_sum = expanded_mask.sum(dim=-1, keepdim=True)
    attn_weights = torch.where(
        mask_sum > 0,
        torch.softmax(scores, dim=-1),
        torch.zeros_like(scores)
    )
    
    output = torch.matmul(attn_weights, v)
    return output

if __name__ == "__main__":
    for i in range(6, 12):
        for d_model in [32, 64, 128]:
            unit_test(2 ** i, d_model)
    print("pass all unit test")
    print("-----------------------------------------")    
    
    benchmark.run(show_plots=False, print_data=True)
