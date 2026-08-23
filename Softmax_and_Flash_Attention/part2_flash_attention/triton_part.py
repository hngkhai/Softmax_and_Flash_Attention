import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()


##
# BLOCK_M : The number of query rows we process in one tile.
# Each Triton program handles a block of queries of shape [BLOCK_M, d_model].

# BLOCK_N : The number of key/value rows we process in one tile inside the inner loop.
# For a given query block, we don't process all keys/values at once.
# Instead, we loop over the key/value sequence in chunks of size [BLOCK_N, d_model].

@triton.jit
def flash_attention_v1(
    q_ptr, k_ptr, v_ptr, o_ptr, scale_factor,
    seq_len: tl.constexpr, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    ## offs_m -> shape [Block M]
    ## for eg, offs_m = [0,1,2,3]
    ## later, if offs_m[:,None] will give [[0], [1], [2],[3]] 
    ## later, offs_m[:, None] * 5 will give [[0],[5], [10],[15]]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    

    ## offs-d -> shape[d_model]
    offs_d = tl.arange(0, d_model) 

    ## offs_m[:, None] - > shape [Block_M,1]
    ## offs_d[None, :] - > shape [1,d_model]   
    ## for eg, if offs_d = [0,1,2] will give [[0,1,2]]
    ## therefore, offs_m[:, None] * stride_qm + offs_d[None, :] will give 
    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :]
    # offs_m:                [4]                →  [0, 1, 2, 3]
    # offs_m[:, None]:       [4, 1]             →  [[0], [1], [2], [3]]
    # * stride_qm:           [4, 1]             →  [[0], [5], [10], [15]]
    # offs_d:                [3]                →  [0, 1, 2]
    # offs_d[None, :]:       [1, 3]             →  [[0, 1, 2]]
    # (4,1) + (1,3) →        [4, 3]             →  [[0,1,2], [5,6,7], [10,11,12], [15,16,17]]
    # + q_ptr:               [4, 3] pointers    →  final addresses

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0).to(tl.float32)


    ## accumulator for the numerator
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)

    ## accumulator for the denominator
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    ## runing maximum of attention scores
    m = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_ptr + offs_n[:, None] * stride_km + offs_d[None, :]
        v_ptrs = v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :]

        k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len), other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len), other=0.0).to(tl.float32)

        s = tl.dot(q, tl.trans(k)) * scale_factor  # [BM, BN]

        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m, m_ij)
  
        alpha = tl.exp(m - m_new)                 # [BM]
        p = tl.exp(s - m_new[:, None])            # [BM, BN]
        l_new = alpha * l + tl.sum(p, axis=1)     # [BM]
        acc = alpha[:, None] * acc + tl.dot(p, v) # [BM, D]
        l = l_new
        m = m_new

    out = acc / l[:, None]
    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, out, mask=(offs_m[:, None] < seq_len))


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def call_flash_attention_v1(q, k, v):
    assert q.shape == k.shape == v.shape, "Input shapes must match"
    assert q.dim() == 2, "Only support 2D input: (seq_len, d_model)"
    seq_len, d_model = q.shape
    o = torch.empty_like(q)
    # Your code here
    if d_model == 256:
        BLOCK_M = 32
        if seq_len <= 256:
            BLOCK_M = 16
        elif seq_len <= 8:
            BLOCK_M = 8
        elif seq_len <=4:
            BLOCK_M = 2
    else:
        BLOCK_M = 64

    
    # Optimize block sizes based on device properties
    max_smem_per_block = SIZE_SMEM // 2  # Conservative estimate
    bytes_per_element = 4  # float32
    smem_limit = SIZE_SMEM 
    safety = 0.9
    max_bn = int((smem_limit * safety) // (2 * d_model * 4))
    max_bn = max(16, min(128, max_bn))
    BLOCK_N = 1 << (max_bn.bit_length() - 1)
    
    # Calculate optimal BLOCK_M and BLOCK_N
    # Memory usage: BLOCK_M * d_model + BLOCK_N * d_model + BLOCK_M * BLOCK_N
    max_block_size = int((max_smem_per_block / bytes_per_element / (2 * d_model + 64)) ** 0.5)
    #BLOCK_M = min(64, max(32, max_block_size))
    
    # Ensure block sizes are powers of 2 for better performance
    #BLOCK_M = 2 ** int(BLOCK_M.bit_length() - 1)
    BLOCK_N = 2 ** int(BLOCK_N.bit_length() - 1)
    
    # Grid size optimized for SM utilization
    grid = (triton.cdiv(seq_len, BLOCK_M),)

    scale_factor = 1.0/(d_model**0.5)

    if d_model < 256:
        num_warps = 8
    else:
        num_warps = 4

    if seq_len < 256:
        num_stages = 3
    else:
        num_stages = 2
    
    # Launch kernel
    flash_attention_v1[grid](
        q, k, v, o, scale_factor,
        seq_len, d_model,
        q.stride(0), k.stride(0), v.stride(0), o.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps = num_warps,
    )
    
    return o

def pytorch_attention(q, k, v):
        d_model = q.shape[-1]
        scale_factor = 1.0 / (d_model**0.5)
        scores = q @ k.T * scale_factor
        softmax = torch.softmax(scores, dim=1)
        attention = softmax @ v
        return attention

def benchmark_internal(seq_len, d_model, provider):
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    if provider == 'flash_attention':
        fn = lambda: call_flash_attention_v1(q, k, v)
    elif provider == 'pytorch':
        fn = lambda: pytorch_attention(q, k, v)

    return triton.testing.do_bench(fn)

def benchmark():
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=[2 ** i for i in range(2, 13)],
            line_arg='d_model',
            line_vals=[64, 128, 256],
            line_names=['d=64', 'd=128', 'd=256'],
            styles=[('blue', '-'), ('green', '--'), ('red', '-.')],
            ylabel="Latency (ms)",
            plot_name="flash-attention-performance",
            args={'provider': 'flash_attention'},
        )
    )
    def bench_flash(seq_len, d_model, provider):
        return benchmark_internal(seq_len, d_model, provider)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['seq_len'],
            x_vals=[2 ** i for i in range(2, 13)],
            line_arg='d_model',
            line_vals=[64, 128, 256],
            line_names=['d=64', 'd=128', 'd=256'],
            styles=[('orange', '-'), ('purple', '--'), ('brown', '-.')],
            ylabel="Latency (ms)",
            plot_name="pytorch-attention-performance",
            args={'provider': 'pytorch'},
        )
    )
    def bench_pytorch(seq_len, d_model, provider):
        return benchmark_internal(seq_len, d_model, provider)

    bench_flash.run(show_plots=False, print_data=True)
    bench_pytorch.run(show_plots=False, print_data=True)
    
    
def unit_test(seq_len, d_model):
    torch.manual_seed(0)
    q = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    k = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)
    v = torch.randn(seq_len, d_model, device=DEVICE, dtype=torch.float32)

    o_triton = call_flash_attention_v1(q, k, v)
    o_torch = pytorch_attention(q, k, v)

    assert torch.allclose(o_triton, o_torch, atol=1e-3, rtol=1e-3), (o_triton, o_torch)
    print(f"Attention output correct for seq_len={seq_len}, d_model={d_model}!")

if __name__ == "__main__":
    for i in range(8, 12):
        for d_model in [32, 64, 128]:
            unit_test(2 ** i, d_model)
    print("pass all unit test")
    print("-----------------------------------------")   
    benchmark()