"""
简化版 FlashAttention for Prefill 阶段
处理长序列的注意力计算
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, seq_len, num_heads, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    简化版FlashAttention Prefill Kernel
    
    Args:
        q_ptr: [batch, num_heads, seq_len, head_dim]
        k_ptr: [batch, num_kv_heads, seq_len, head_dim]
        v_ptr: [batch, num_kv_heads, seq_len, head_dim]
        out_ptr: 输出
    """
    # 获取程序ID
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(2)  # 序列块M
    
    # 计算当前Q块的偏移
    start_m = pid_m * BLOCK_M
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # 计算当前Q块的范围
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # 加载Q块
    q_ptrs = (q_ptr + 
              pid_b * stride_qb + 
              pid_h * stride_qh +
              offs_m[:, None] * stride_qs +
              offs_d[None, :] * stride_qd)
    
    # 掩码
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim
    
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale
    
    # 遍历K,V序列块
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # 加载K块
        k_ptrs = (k_ptr + 
                  pid_b * stride_kb + 
                  pid_h * stride_kh +
                  offs_n[None, :] * stride_ks +
                  offs_d[:, None] * stride_kd)
        
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        
        # 计算S = Q @ K^T
        s = tl.dot(q, k)
        
        # 应用掩码（causal mask）
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        s = tl.where(causal_mask & mask_n[None, :], s, float('-inf'))
        
        # Online Softmax
        m_new = tl.maximum(m, tl.max(s, axis=1))
        
        # 修正累加器
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        
        acc = acc * alpha[:, None]
        
        # 加载V块
        v_ptrs = (v_ptr + 
                  pid_b * stride_vb + 
                  pid_h * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 累加
        acc += tl.dot(p, v)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
    
    # 归一化
    acc = acc / l[:, None]
    
    # 存储输出
    out_ptrs = (out_ptr + 
                pid_b * stride_ob + 
                pid_h * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    FlashAttention Prefill接口
    
    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: [batch, num_kv_heads, seq_len, head_dim]
    
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 确保连续
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # 分配输出
    output = torch.empty_like(q)
    
    # 计算scale
    scale = head_dim ** -0.5
    
    # 配置
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_D = min(BLOCK_D, 128)
    
    # 启动kernel
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))
    
    flash_attn_prefill_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return output


# 参考实现
def attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch参考实现
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 输出
    output = torch.matmul(attn_weights, v)
    
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FlashAttention Prefill...")
    
    # 测试配置
    batch_size = 1
    num_heads = 24
    seq_len = 128
    head_dim = 128
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # PyTorch参考
    with torch.no_grad():
        ref_output = attention_pytorch(q, k, v)
    
    # Triton实现
    triton_output = flash_attention_prefill(q, k, v)
    
    # 对比
    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✓ FlashAttention Prefill test PASSED!")
    else:
        print("✗ FlashAttention Prefill test FAILED!")