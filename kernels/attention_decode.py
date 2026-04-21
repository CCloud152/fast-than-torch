"""
简化版 FlashAttention for Decode 阶段
处理单token生成时的注意力计算（KV Cache优化）
"""

import torch
import triton
import triton.language as tl

def ref_pytorch(
    q: torch.tensor,
    k_cache: torch.tensor,
    v_cache: torch.tensor,
    kv_len: int    
):
    batch_size, num_heads, head_dim = q.shape
    _, num_kv_heads, _, _ = k_cache.shape

    k = k_cache[:, :, :kv_len, :]
    v = v_cache[:, :, :kv_len, :]

    if num_kv_heads < num_heads:
        num_repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(num_repeat, dim=1)
        v = v.repeat_interleave(num_repeat, dim=1)

    scores = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (head_dim ** 0.5)
    scores = scores.squeeze(2)

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights.unsqueeze(2), v).squeeze(2)

    return output

@triton.jit
def flash_attn_decode_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
    batch_size, num_heads, num_kv_heads, head_dim, kv_len,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    scale: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    num_heads_per_kv = num_heads // num_kv_heads
    pid_h_kv = pid_h // num_heads_per_kv

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=mask_d, other=0.0)
    q = q * scale
    
    m = float('-inf')
    l = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    for start_n in range(0, kv_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        
        k_ptrs = (k_cache_ptr + 
                  pid_b * stride_kb + 
                  pid_h_kv * stride_kh +
                  offs_n[:, None] * stride_ks +
                  offs_d[None, :] * stride_kd)
        
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        s = tl.sum(q[None, :] * k, axis=1)
        
        m_new = tl.maximum(m, tl.max(s))
        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new)
        
        acc = acc * alpha
        
        v_ptrs = (v_cache_ptr + 
                  pid_b * stride_vb + 
                  pid_h_kv * stride_vh +
                  offs_n[:, None] * stride_vs +
                  offs_d[None, :] * stride_vd)
        
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        acc += tl.sum(p[:, None] * v, axis=0)
        l = l * alpha + tl.sum(p)
        m = m_new
    
    acc = acc / l
    
    out_ptrs = out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od
    tl.store(out_ptrs, acc, mask=mask_d)


def flash_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_len: int,
) -> torch.Tensor:
    batch_size, num_heads, head_dim = q.shape
    _, num_kv_heads, _, _ = k_cache.shape
    
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    
    output = torch.empty_like(q)
    
    scale = head_dim ** -0.5
    
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_D = min(BLOCK_D, 128)
    BLOCK_N = 64
    
    grid = (batch_size, num_heads)
    
    flash_attn_decode_kernel[grid](
        q, k_cache, v_cache, output,
        batch_size, num_heads, num_kv_heads, head_dim, kv_len,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        scale=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
    )
    
    return output


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FlashAttention Decode...")
    
    # 测试配置
    batch_size = 1
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 2048
    kv_len = 512  # 假设已经生成了512个token
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(batch_size, num_kv_heads, max_seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # PyTorch参考
    with torch.no_grad():
        ref_output = ref_pytorch(q, k_cache, v_cache, kv_len)
    
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