"""
融合 RoPE (Rotary Position Embedding) Triton Kernel
功能: sin/cos计算 + 旋转操作
"""

import torch
import triton
import triton.language as tl

def ref_pytorch(
    q: torch.tensor,
    k: torch.tensor,
    cos: torch.tensor,
    sin: torch.tensor,
):
    q_x1 = q[..., ::2]
    q_x2 = q[..., 1::2]
    k_x1 = k[..., ::2]
    k_x2 = k[..., 1::2]

    cos = cos.unsqueeze(1).unsqueeze(0)
    sin = sin.unsqueeze(1).unsqueeze(0)
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    q_out[..., ::2] = q_x1 * cos - q_x2 * sin
    q_out[..., 1::2] = q_x1 * sin + q_x2 * cos

    k_out[..., ::2] = k_x1 * cos - k_x2 * sin
    k_out[..., 1::2] = k_x1 * sin + k_x2 * cos
    
    return q_out, k_out

@triton.jit
def rope_fused_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr,
    out_q_ptr, out_k_ptr,
    seq_len, num_heads, num_kv_heads, head_dim,
    stride_qb, stride_qs, stride_qh, 
    stride_kb, stride_ks, stride_kh,
    BLOCK_SIZE: tl.constexpr 
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    q_offs = pid_b * stride_qb + pid_s * stride_qs + pid_h * stride_qh
    q_base = q_ptr + q_offs
    k_offs = pid_b * stride_kb + pid_s * stride_ks + pid_h * stride_kh
    k_base = k_ptr + k_offs

    k_valid = pid_h < num_kv_heads
    cos_sin_offset = pid_s * (head_dim // 2) # [seq_len, head_dim // 2]

    for i in range(0, head_dim // 2, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < (head_dim // 2)

        cos = tl.load(cos_ptr + cos_sin_offset + offs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + cos_sin_offset + offs, mask=mask, other=0.0)

        q_x1 = tl.load(q_base + 2 * offs, mask=mask, other=0.0)
        q_x2 = tl.load(q_base + 2 * offs + 1, mask=mask, other=0.0)
        q_y1 = q_x1 * cos - q_x2 * sin
        q_y2 = q_x1 * sin + q_x2 * cos
        tl.store(out_q_ptr + q_offs + 2 * offs, q_y1, mask=mask)
        tl.store(out_q_ptr + q_offs + 2 * offs + 1, q_y2, mask=mask)

        k_mask = mask & k_valid
        k_x1 = tl.load(k_base + 2 * offs, mask=k_mask, other=0.0)
        k_x2 = tl.load(k_base + 2 * offs + 1, mask=k_mask, other=0.0)
        k_y1 = k_x1 * cos - k_x2 * sin
        k_y2 = k_x1 * sin + k_x2 * cos
        tl.store(out_k_ptr + k_offs + 2 * offs, k_y1, mask=k_mask)
        tl.store(out_k_ptr + k_offs + 2 * offs + 1, k_y2, mask=k_mask)

def rope_fused(
    q: torch.tensor,
    k: torch.tensor,
    cos: torch.tensor,
    sin: torch.tensor,
):
    batch_size, seq_len, num_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = k.shape

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)
    
    q = q.contiguous()
    k = k.contiguous()

    BLOCK_SIZE = min(128, head_dim // 2)
    grid = (batch_size, seq_len, max(num_heads, num_kv_heads))

    rope_fused_kernel[grid](
        q, k, cos, sin,
        out_q, out_k,
        seq_len, num_heads, num_kv_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        BLOCK_SIZE=BLOCK_SIZE 
    )

    return out_q, out_k

def precompute_rope_rotary_cache(
    max_seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: str = "cuda",
):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    
    t = torch.arange(max_seq_len, device=device)

    freqs = torch.outer(t, freqs)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin

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
    q_ref, k_ref = ref_pytorch(q, k, cos, sin)
    
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