"""
融合 FFN (Feed-Forward Network) Triton Kernel
功能: matmul计算 + SiLU/SwiGLU激活
"""

import torch
import triton
import triton.language as tl

def ref_pytorch(
    x: torch.tensor,
    w_gate: torch.tensor,
    w_up: torch.tensor,
    w_down: torch.tensor,
):
    gate = torch.matmul(x, w_gate.t())

    up = torch.matmul(x, w_up.t())

    hidden = torch.nn.functional.silu(gate) * up

    output = torch.matmul(hidden, w_down)
    return output

@triton.jit
def silu_trion(x):
    return x * tl.sigmoid(x)

@triton.jit
def swiglu_triton(x, y):
    return silu_trion(x) * y

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key = ['M', 'K', 'N'],
)
@triton.jit
def ffn_fused_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    w3_ptr,
    out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_w1_k, stride_w1_n,
    stride_w2_k, stride_w2_n,
    stride_w3_n, stride_w3_k,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k_inner = k_start + tl.arange(0, BLOCK_K)
            mask_k_inner = offs_k_inner < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k_inner[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k_inner[None, :], other=0.0)

            w1_ptrs = w1_ptr + offs_k_inner[:, None] * stride_w1_k + offs_n[None, :] * stride_w1_n
            w1_tile = tl.load(w1_ptrs, mask=mask_k_inner[:, None] & mask_n[None, :], other=0.0)

            w2_ptrs = w2_ptr + offs_k_inner[:, None] * stride_w2_k + offs_n[None, :] * stride_w2_n
            w2_tile = tl.load(w2_ptrs, mask=mask_k_inner[:, None] & mask_n[None, :], other=0.0)

            gate += tl.dot(x_tile, w1_tile)
            up += tl.dot(x_tile, w2_tile)

        hidden = swiglu_triton(gate, up)

        w3_ptrs = w3_ptr + offs_n[:, None] * stride_w3_n + offs_k[None, :] * stride_w3_k 
        w3_tile = tl.load(w3_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0)

        acc += tl.dot(hidden.to(tl.float16), w3_tile)
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask_m[:, None] & mask_k[None, :])


def ffn_fused_swiglu(
    x: torch.tensor,
    w_gate: torch.tensor,
    w_up: torch.tensor,
    w_down: torch.tensor,
):
    original_shape = x.shape
    if x.dim() == 3:
        x = x.view(-1, x.shape[-1])
    
    assert x.dim() == 2, f"Expected 2D tensor [M, K], got {x.dim()}D"
    M, K = x.shape
    N = w_gate.shape[0]

    x = x.contiguous()
    w_gate_t = w_gate.t().contiguous()
    w_up_t = w_up.t().contiguous()
    w_down_c = w_down.contiguous()

    output = torch.empty_like(x)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(K, META['BLOCK_K'])
    )

    ffn_fused_kernel[grid](
        x,
        w_gate_t, w_up_t, w_down_c,
        output,
        M, K, N,
        x.stride(0), x.stride(1),
        w_gate_t.stride(0), w_gate_t.stride(1),
        w_up_t.stride(0), w_up_t.stride(1),
        w_down_c.stride(0), w_down_c.stride(1),
        output.stride(0), output.stride(1),
    )

    if len(original_shape) == 3:
        output = output.view(original_shape)

    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FFN SwiGLU Triton Kernel...")
    
    batch_size, seq_len = 2, 16
    hidden_size = 3072
    intermediate_size = 8192
    M = batch_size * seq_len
    
    torch.manual_seed(42)
    x = torch.randn(M, hidden_size, device="cuda", dtype=torch.float16)
    w_gate = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.01
    w_up = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.01
    w_down = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.01
    
    with torch.no_grad():
        ref_output = ref_pytorch(x, w_gate, w_up, w_down)
    
    triton_output = ffn_fused_swiglu(x, w_gate, w_up, w_down)
    
    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✓ FFN SwiGLU test PASSED!")
    else:
        print("✗ FFN SwiGLU test FAILED!")