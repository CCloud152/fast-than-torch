"""
简化版 FlashAttention for Prefill 阶段
处理长序列的注意力计算
"""

import torch
import triton
import triton.language as tl

def ref_pytorch(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
):
    batch_size, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _, = k.shape

    if num_kv_heads < num_heads:
        num_repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(num_repeat, dim=1)
        v = v.repeat_interleave(num_repeat, dim=1)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)

    return output

@triton.jit
def flash_attn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, seq_len, num_heads, num_kv_heads, head_dim,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    num_heads_per_kv = num_heads // num_kv_heads
    pid_h_kv = pid_h // num_heads_per_kv

    start_m = pid_m * BLOCK_M

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (q_ptr + 
              pid_b * stride_qb + 
              pid_h * stride_qh + 
              offs_m[:, None] * stride_qs + 
              offs_d[None, :] * stride_qd)
    
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim

    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len

        k_ptrs = (k_ptr +
                  pid_b * stride_kb +
                  pid_h_kv * stride_kh + 
                  offs_d[:, None] * stride_kd +
                  offs_n[None, :] * stride_ks)
        
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

        s = tl.dot(q, k)

        casual_mask = (offs_m[:, None] >= offs_n[None, :]) & mask_m[:, None] & mask_n[None, :]
        s = tl.where(casual_mask, s, float('-inf'))

        row_max = tl.max(s, axis = 1)
        m_new = tl.maximum(m, row_max)

        alpha = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])

        acc = acc * alpha[:, None]

        v_ptrs = (v_ptr +
                  pid_b * stride_vb +
                  pid_h_kv * stride_vh + 
                  offs_n[:, None] * stride_vs + 
                  offs_d[None, :] * stride_vd)
        
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new

    l_safe = tl.where(l == 0.0, 1.0, l)
    acc = acc / l_safe[:, None]

    acc = tl.where(mask_m[:, None] & mask_d[None, :], acc, 0.0)

    out_ptrs = (out_ptr +
                pid_b * stride_ob +
                pid_h * stride_oh +
                offs_m[:, None] * stride_os +
                offs_d[None, :] * stride_od)
    
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])
    
def flash_attention_prefill(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
):
    batch_size, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    output = torch.empty_like(q)

    scale = head_dim ** -0.5

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_D = min(BLOCK_D, 128)

    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_M))

    flash_attn_prefill_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return output

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing FlashAttention Prefill...")
    
    # 测试配置
    batch_size = 1
    num_heads = 24
    num_kv_heads = 8
    seq_len = 128
    head_dim = 128
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    
    # PyTorch参考
    with torch.no_grad():
        ref_output = ref_pytorch(q, k, v)
    
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