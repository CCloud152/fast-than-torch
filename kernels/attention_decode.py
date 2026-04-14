"""
简化版 FlashAttention for Decode 阶段
处理单token生成时的注意力计算（KV Cache优化）
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    batch_size, num_heads, head_dim, kv_len,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    FlashAttention Decode Kernel
    
    Decode阶段特点:
    - Query只有1个token
    - Key/Value来自KV Cache（可能很长）
    
    Args:
        q_ptr: [batch, num_heads, head_dim] - 只有1个token
        k_cache_ptr: [batch, num_kv_heads, max_seq_len, head_dim]
        v_cache_ptr: [batch, num_kv_heads, max_seq_len, head_dim]
    """
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # head
    
    # 加载Query（单个token）
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0)
    q = q * scale
    
    # 初始化softmax统计
    m = float('-inf')
    l = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # 遍历KV Cache
    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        
        # 加载K
        k_ptrs = (k_cache_ptr + 
                  pid_b * stride_kb + 
                  pid_h * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 计算注意力分数 s = q @ k^T
        # q: [head_dim], k: [BLOCK_N, head_dim] -> s: [BLOCK_N]
        s = tl.sum(q[None, :] * k, axis=1)
        
        # Online softmax
        m_new = tl.maximum(m, tl.max(s))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        
        # 修正累加器
        acc = acc * alpha
        
        # 加载V
        v_ptrs = (v_cache_ptr + 
                  pid_b * stride_vb + 
                  pid_h * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 累加
        acc += tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p)
        m = m_new
    
    # 归一化
    acc = acc / l
    
    # 存储输出
    out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc, mask=mask_d)


def flash_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_len: int,
) -> torch.Tensor:
    """
    FlashAttention Decode接口
    
    Args:
        q: [batch, num_heads, head_dim] - 单个query
        k_cache: [batch, num_kv_heads, max_seq_len, head_dim]
        v_cache: [batch, num_kv_heads, max_seq_len, head_dim]
        kv_len: 当前KV Cache长度
    
    Returns:
        output: [batch, num_heads, head_dim]
    """
    batch_size, num_heads, head_dim = q.shape
    
    # 确保连续
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    
    # 分配输出
    output = torch.empty_like(q)
    
    # 计算scale
    scale = head_dim ** -0.5
    
    # 配置
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_D = min(BLOCK_D, 128)
    BLOCK_N = 64
    
    # 启动kernel
    grid = (batch_size, num_heads)
    
    flash_attn_decode_kernel[grid](
        q, k_cache, v_cache, output,
        batch_size, num_heads, head_dim, kv_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        scale=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    
    return output


# PyTorch参考实现
def attention_decode_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_len: int,
) -> torch.Tensor:
    """
    PyTorch参考实现
    """
    batch_size, num_heads, head_dim = q.shape
    
    # 截取有效的KV
    k = k_cache[:, :, :kv_len, :]  # [batch, num_heads, kv_len, head_dim]
    v = v_cache[:, :, :kv_len, :]
    
    # 计算注意力分数
    # q: [batch, num_heads, 1, head_dim]
    # k: [batch, num_heads, kv_len, head_dim]
    scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (head_dim ** 0.5)
    scores = scores.squeeze(2)  # [batch, num_heads, kv_len]
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 计算输出
    # attn_weights: [batch, num_heads, kv_len]
    # v: [batch, num_heads, kv_len, head_dim]
    output = torch.matmul(attn_weights.unsqueeze(2), v).squeeze(2)
    
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FlashAttention Decode...")
    
    # 测试配置
    batch_size = 1
    num_heads = 24
    head_dim = 128
    max_seq_len = 2048
    kv_len = 512  # 假设已经生成了512个token
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(batch_size, num_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(batch_size, num_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # PyTorch参考
    with torch.no_grad():
        ref_output = attention_decode_pytorch(q, k_cache, v_cache, kv_len)
    
    # Triton实现
    triton_output = flash_attention_decode(q, k_cache, v_cache, kv_len)
    
    # 对比
    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-2:
        print("✓ FlashAttention Decode test PASSED!")
    else:
        print("✗ FlashAttention Decode test FAILED!")