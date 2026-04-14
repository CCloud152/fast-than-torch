"""
融合 RMSNorm Triton Kernel
功能: RMSNorm + Residual + Activation
"""


import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_fused_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    residual_ptr,  # 可选的残差输入
    has_residual: tl.constexpr,
    eps,
    n_elements,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合RMSNorm Kernel
    
    Args:
        x_ptr: 输入张量 [M, N]
        weight_ptr: 权重 [N]
        out_ptr: 输出 [M, N]
        residual_ptr: 残差输入 [M, N] (可选)
        has_residual: 是否有残差
        eps: 防止除零的小常数
        n_elements: N维度大小
        stride_row: 行步长
    """
    # 获取当前行的索引
    row_idx = tl.program_id(0)
    
    # 计算当前行的偏移
    row_start = x_ptr + row_idx * stride_row
    
    # 计算当前行的有效列数
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_elements
    
    # 加载输入
    x = tl.load(row_start + cols, mask=mask, other=0.0)
    
    # 如果有残差，先加残差
    if has_residual:
        res_ptr = residual_ptr + row_idx * stride_row
        residual = tl.load(res_ptr + cols, mask=mask, other=0.0)
        x = x + residual
    
    # 计算RMS: sqrt(mean(x^2) + eps)
    x_float32 = x.to(tl.float32)
    square_sum = tl.sum(x_float32 * x_float32, axis=0)
    mean_square = square_sum / n_elements
    rms = tl.sqrt(mean_square + eps)
    
    # 归一化
    x_norm = x_float32 / rms
    
    # 加载权重并应用
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    weight = weight.to(tl.float32)
    output = x_norm * weight
    
    # 存储结果
    out_row = out_ptr + row_idx * stride_row
    tl.store(out_row + cols, output.to(x.dtype), mask=mask)


def rms_norm_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    residual: torch.Tensor = None,
) -> torch.Tensor:
    """
    融合RMSNorm函数接口
    
    Args:
        x: 输入张量 [M, N]
        weight: 权重 [N]
        eps: 防止除零的小常数
        residual: 可选的残差张量 [M, N]
    
    Returns:
        output: 归一化后的张量 [M, N]
    """
    # 确保输入是连续的
    x = x.contiguous()
    weight = weight.contiguous()
    if residual is not None:
        residual = residual.contiguous()
    
    # 获取形状
    M, N = x.shape
    
    # 分配输出
    output = torch.empty_like(x)
    
    # 选择block大小（必须是2的幂次）
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # 限制最大block大小
    
    # 启动kernel
    grid = (M,)
    rms_norm_fused_kernel[grid](
        x,
        weight,
        output,
        residual if residual is not None else x,  # 占位符
        residual is not None,
        eps,
        N,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# 测试代码
if __name__ == "__main__":
    import torch.nn.functional as F
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing RMSNorm Triton Kernel...")
    
    # 测试配置
    batch_size, seq_len, hidden_size = 2, 16, 3072
    M = batch_size * seq_len
    
    # 创建随机输入
    torch.manual_seed(42)
    x = torch.randn(M, hidden_size, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_size, device="cuda", dtype=torch.float16)
    
    # PyTorch参考实现
    def rms_norm_pytorch(x, weight, eps=1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x
    
    # 对比测试
    ref_output = rms_norm_pytorch(x, weight)
    triton_output = rms_norm_fused(x, weight)
    
    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()
    
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("✓ RMSNorm test PASSED!")
    else:
        print("✗ RMSNorm test FAILED!")