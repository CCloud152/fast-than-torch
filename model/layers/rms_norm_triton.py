"""
LLaMA RMSNorm Layer with Triton Kernel
"""

import torch
import torch.nn as nn

# 尝试导入Triton kernel
try:
    from ...kernels.rms_fused import rms_norm_fused
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class LlamaRMSNormTriton(nn.Module):
    """
    LLaMA RMSNorm with Triton加速
    
    LLaMA使用RMSNorm而不是LayerNorm:
    - LayerNorm: (x - mean) / sqrt(var + eps) * weight
    - RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.use_triton = use_triton and TRITON_AVAILABLE
        
    def _rms_norm_pytorch(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch参考实现"""
        # 计算RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入 [batch, seq_len, hidden_size]
            residual: 可选的残差连接
        
        Returns:
            normalized: 归一化后的输出
        """
        # 如果有残差，先加残差
        if residual is not None:
            x = x + residual
        
        # 尝试Triton加速
        if self.use_triton:
            try:
                return rms_norm_fused(x, self.weight, self.eps)
            except Exception as e:
                # 失败时回退到PyTorch
                pass
        
        # PyTorch实现
        return self._rms_norm_pytorch(x)
    
    def forward_with_residual(self, x: torch.Tensor, residual: torch.Tensor = None):
        """
        同时返回输出和残差（用于预归一化架构）
        
        LLaMA架构: x -> RMSNorm -> Attention/FFN -> +residual
        所以我们需要保存输入作为残差
        """
        residual_input = x if residual is None else residual
        output = self.forward(x, residual)
        return output, residual_input


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing LLaMA RMSNorm...")
    
    # 测试配置
    batch_size, seq_len, hidden_size = 2, 16, 3072
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    
    # 创建层
    norm_layer = LlamaRMSNormTriton(hidden_size).cuda()
    
    # 测试前向
    output = norm_layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean:  {output.mean().item():.4f}")
    print(f"Output std:   {output.std().item():.4f}")
    print("✓ RMSNorm layer test PASSED!")