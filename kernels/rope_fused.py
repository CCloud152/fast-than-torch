"""
融合 RoPE (Rotary Position Embedding) Triton Kernel
功能: sin/cos计算 + 旋转操作
"""


import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_fused_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    seq_len,
    num_heads,
    head_dim,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合RoPE Kernel
    
    Args:
        q_ptr: Query张量 [batch, seq_len, num_heads, head_dim]
        k_ptr: Key张量 [batch, seq_len, num_kv_heads, head_dim]
        cos_ptr: Cos缓存 [seq_len, head_dim//2]
        sin_ptr: Sin缓存 [seq_len, head_dim//2]
        out_q_ptr: 输出Query
        out_k_ptr: 输出Key
    """
    # 获取当前位置
    pid_b = tl.program_id(0)  # batch维度
    pid_s = tl.program_id(1)  # seq维度
    pid_h = tl.program_id(2)  # head维度
    
    # 计算偏移
    q_base = q_ptr + pid_b * stride_qb + pid_s * stride_qs + pid_h * stride_qh
    k_base = k_ptr + pid_b * stride_kb + pid_s * stride_ks + pid_h * stride_kh
    
    # 计算旋转偏移
    cos_sin_offset = pid_s * (head_dim // 2)
    
    # 处理每一对维度
    for i in range(0, head_dim // 2, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < (head_dim // 2)
        
        # 加载cos和sin
        cos = tl.load(cos_ptr + cos_sin_offset + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + cos_sin_offset + offs, mask=mask, other=0.0)
        
        # 加载原始q, k的两个部分
        # x1 = q[..., ::2], x2 = q[..., 1::2]
        q_x1 = tl.load(q_base + 2 * offs, mask=mask, other=0.0)
        q_x2 = tl.load(q_base + 2 * offs + 1, mask=mask, other=0.0)
        
        k_x1 = tl.load(k_base + 2 * offs, mask=mask, other=0.0)
        k_x2 = tl.load(k_base + 2 * offs + 1, mask=mask, other=0.0)
        
        # 旋转: [x1, x2] @ [[cos, -sin], [sin, cos]]
        # y1 = x1 * cos - x2 * sin
        # y2 = x1 * sin + x2 * cos
        q_y1 = q_x1 * cos - q_x2 * sin
        q_y2 = q_x1 * sin + q_x2 * cos
        
        k_y1 = k_x1 * cos - k_x2 * sin
        k_y2 = k_x1 * sin + k_x2 * cos
        
        # 存储结果
        tl.store(out_q_ptr + q_base - q_ptr + 2 * offs, q_y1, mask=mask)
        tl.store(out_q_ptr + q_base - q_ptr + 2 * offs + 1, q_y2, mask=mask)
        
        tl.store(out_k_ptr + k_base - k_ptr + 2 * offs, k_y1, mask=mask)
        tl.store(out_k_ptr + k_base - k_ptr + 2 * offs + 1, k_y2, mask=mask)


def precompute_rope_rotary_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    预计算RoPE的sin/cos缓存
    
    Args:
        max_seq_len: 最大序列长度
        head_dim: 注意力头维度
        theta: 旋转基频
        device: 设备
    
    Returns:
        cos, sin缓存
    """
    # 计算频率
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    # 计算位置
    t = torch.arange(max_seq_len, device=device)
    
    # 外积得到 [seq_len, head_dim//2]
    freqs = torch.outer(t, freqs)
    
    # 计算cos和sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    
    return cos, sin


def rope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合RoPE函数接口
    
    Args:
        q: Query [batch, seq_len, num_heads, head_dim]
        k: Key [batch, seq_len, num_kv_heads, head_dim]
        cos: Cos缓存 [seq_len, head_dim//2]
        sin: Sin缓存 [seq_len, head_dim//2]
    
    Returns:
        q_rotated, k_rotated
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape
    
    # 分配输出
    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    
    # 确保输入连续
    q = q.contiguous()
    k = k.contiguous()
    
    # 启动kernel
    BLOCK_SIZE = min(128, head_dim // 2)
    grid = (batch_size, seq_len, max(num_heads, num_kv_heads))
    
    rope_fused_kernel[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        seq_len,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_q, out_k


# 更高效的简化版本（使用PyTorch原生操作）
def rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch参考实现
    
    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos: [seq_len, head_dim//2]
        sin: [seq_len, head_dim//2]
    """
    # 重塑以便处理
    batch, seq_len, num_heads, head_dim = q.shape
    
    # 分离偶数和奇数维度
    q_x1 = q[..., ::2]  # [batch, seq_len, num_heads, head_dim//2]
    q_x2 = q[..., 1::2]
    k_x1 = k[..., ::2]
    k_x2 = k[..., 1::2]
    
    # 扩展cos和sin维度
    cos = cos.unsqueeze(1).unsqueeze(0)  # [1, seq_len, 1, head_dim//2]
    sin = sin.unsqueeze(1).unsqueeze(0)
    
    # 应用旋转
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    q_out[..., ::2] = q_x1 * cos - q_x2 * sin
    q_out[..., 1::2] = q_x1 * sin + q_x2 * cos
    
    k_out[..., ::2] = k_x1 * cos - k_x2 * sin
    k_out[..., 1::2] = k_x1 * sin + k_x2 * cos
    
    return q_out, k_out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing RoPE Triton Kernel...")
    
    # 测试配置
    batch_size = 2
    seq_len = 16
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    
    # 创建随机输入
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    
    # 预计算缓存
    cos, sin = precompute_rope_rotary_cache(seq_len, head_dim, theta=500000.0)
    
    # PyTorch参考
    q_ref, k_ref = rope_pytorch(q, k, cos, sin)
    
    # Triton实现
    q_tri, k_tri = rope_fused(q, k, cos, sin)
    
    # 对比
    max_diff_q = torch.max(torch.abs(q_ref - q_tri)).item()
    max_diff_k = torch.max(torch.abs(k_ref - k_tri)).item()
    
    print(f"Max diff Q: {max_diff_q:.6f}")
    print(f"Max diff K: {max_diff_k:.6f}")
    
    if max_diff_q < 1e-3 and max_diff_k < 1e-3:
        print("✓ RoPE test PASSED!")
    else:
        print("✗ RoPE test FAILED!")
