"""
融合 FFN (Feed-Forward Network) Triton Kernel
功能: matmul计算 + SiLU/SwiGLU激活
"""

import torch
import triton
import triton.language as tl


@triton.jit
def silu_triton(x):
    """SiLU激活函数: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def swiglu_triton(x, y):
    """SwiGLU: SiLU(x) * y"""
    return silu_triton(x) * y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def ffn_fused_kernel(
    x_ptr,
    w1_ptr,  # gate_proj / w_gate
    w3_ptr,  # up_proj / w_up
    w2_ptr,  # down_proj / w_down
    out_ptr,
    M, K, N,  # M=bs*seq_len, K=hidden_size, N=intermediate_size
    stride_xm, stride_xk,
    stride_w1_n, stride_w1_k,
    stride_w3_n, stride_w3_k,
    stride_w2_k, stride_w2_n,
    stride_om, stride_ok,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    融合FFN Kernel: SwiGLU + Down Projection
    
    LLaMA FFN结构:
    - gate = x @ w1^T  (hidden -> intermediate)
    - up = x @ w3^T    (hidden -> intermediate)
    - hidden = SwiGLU(gate, up) = SiLU(gate) * up
    - out = hidden @ w2^T  (intermediate -> hidden)
    
    融合策略: 不保存中间结果，直接计算最终输出
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # 计算当前tile的偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    
    # 创建掩码
    mask_m = offs_m < M
    mask_k = offs_k < K
    
    # 初始化输出累加器
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    
    # 遍历中间维度N (intermediate_size)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Step 1: 计算 gate = x @ w1^T 的当前tile
        gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # 内层循环遍历K维度
        for k_start in range(0, K, BLOCK_K):
            offs_k_inner = k_start + tl.arange(0, BLOCK_K)
            mask_k_inner = offs_k_inner < K
            
            # 加载x的tile: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k_inner[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k_inner[None, :], other=0.0)
            
            # 加载w1的tile (gate)
            w1_ptrs = w1_ptr + offs_n[:, None] * stride_w1_n + offs_k_inner[None, :] * stride_w1_k
            w1_tile = tl.load(w1_ptrs, mask=mask_n[:, None] & mask_k_inner[None, :], other=0.0)
            
            # 加载w3的tile (up)
            w3_ptrs = w3_ptr + offs_n[:, None] * stride_w3_n + offs_k_inner[None, :] * stride_w3_k
            w3_tile = tl.load(w3_ptrs, mask=mask_n[:, None] & mask_k_inner[None, :], other=0.0)
            
            # 累加矩阵乘
            gate += tl.dot(x_tile, tl.trans(w1_tile))
            up += tl.dot(x_tile, tl.trans(w3_tile))
        
        # Step 2: SwiGLU激活 (融合点)
        # hidden = SiLU(gate) * up
        hidden = swiglu_triton(gate, up)
        
        # Step 3: 累加 Down Projection: out += hidden @ w2
        # w2: [K, N] 即 [hidden_size, intermediate_size]
        w2_ptrs = w2_ptr + offs_k[None, :] * stride_w2_k + offs_n[:, None] * stride_w2_n
        w2_tile = tl.load(w2_ptrs, mask=mask_k[None, :] & mask_n[:, None], other=0.0)
        
        # hidden[BLOCK_M, BLOCK_N] @ w2_tile[BLOCK_N, BLOCK_K]
        acc += tl.dot(hidden, w2_tile)
    
    # 存储结果
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


def ffn_fused_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """
    融合SwiGLU FFN函数接口
    
    Args:
        x: 输入 [M, K] (batch*seq_len, hidden_size)
        w_gate: Gate投影权重 [N, K] (intermediate, hidden)
        w_up: Up投影权重 [N, K] (intermediate, hidden)
        w_down: Down投影权重 [K, N] (hidden, intermediate)
    
    Returns:
        output: [M, K]
    """
    # 处理输入形状
    original_shape = x.shape
    if x.dim() == 3:
        x = x.view(-1, x.shape[-1])
    
    assert x.dim() == 2, f"Expected 2D tensor [M, K], got {x.dim()}D"
    M, K = x.shape
    N = w_gate.shape[0]
    
    # 确保连续内存布局
    x = x.contiguous()
    w_gate_t = w_gate.t().contiguous()  # [N, K]
    w_up_t = w_up.t().contiguous()      # [N, K]
    w_down_c = w_down.contiguous()      # [K, N]
    
    # 分配输出
    output = torch.empty_like(x)
    
    # 启动网格
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(K, META['BLOCK_K'])
    )
    
    # 调用kernel
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
    
    # 恢复形状
    if len(original_shape) == 3:
        output = output.view(original_shape)
    
    return output


# PyTorch参考实现
def ffn_pytorch(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch参考实现
    """
    # Gate projection
    gate = torch.matmul(x, w_gate.t())
    # Up projection
    up = torch.matmul(x, w_up.t())
    # SwiGLU
    hidden = torch.nn.functional.silu(gate) * up
    # Down projection
    output = torch.matmul(hidden, w_down.t())
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FFN SwiGLU Triton Kernel...")
    
    # LLaMA 3.2 3B 配置
    batch_size, seq_len = 2, 16
    hidden_size = 3072
    intermediate_size = 8192
    M = batch_size * seq_len
    
    # 创建随机输入
    torch.manual_seed(42)
    x = torch.randn(M, hidden_size, device="cuda", dtype=torch.float16)
    w_gate = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    w_up = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=torch.float16) * 0.1
    w_down = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=torch.float16) * 0.1
    
    # PyTorch参考
    with torch.no_grad():
        ref_output = ffn_pytorch(x, w_gate, w_up, w_down)
    
    # Triton实现
    triton_output = ffn_fused_swiglu(x, w_gate, w_up, w_down)
    
    # 对比
    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:  # 使用更大的容忍度因为float16
        print("✓ FFN SwiGLU test PASSED!")
    else:
        print("✗ FFN SwiGLU test FAILED!")