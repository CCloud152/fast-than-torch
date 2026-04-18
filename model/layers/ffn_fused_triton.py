"""
LLaMA MLP (FFN) with Triton Kernel
"""

import torch
import torch.nn as nn

# 尝试导入Triton kernel
try:
    from ...kernels.ffn_fused import ffn_fused_swiglu
    TRITON_AVAILABLE = True
except ImportError:
    try:
        from kernels.ffn_fused import ffn_fused_swiglu
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False


class LlamaMLPTriton(nn.Module):
    """
    LLaMA MLP (FFN) with Triton加速
    
    LLaMA使用SwiGLU激活:
    - gate = x @ W_gate
    - up = x @ W_up
    - hidden = SiLU(gate) * up
    - output = hidden @ W_down
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # 三个投影矩阵
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        if self.use_triton:
            try:
                # 使用Triton融合kernel
                return ffn_fused_swiglu(
                    x,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                    self.down_proj.weight,
                )
            except Exception as e:
                # 失败时回退到PyTorch
                pass
        
        # PyTorch实现
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = self.act_fn(gate) * up
        return self.down_proj(hidden)
    
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing LLaMA MLP (Triton Ver)...")
    
    # 测试配置
    batch_size = 2
    seq_len = 16
    hidden_size = 3072
    intermediate_size = 8192 # LLaMA 通常中间层更大
    
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    
    # 创建层
    mlp_layer = LlamaMLPTriton(hidden_size, intermediate_size).cuda()
    
    # 测试前向传播
    output = mlp_layer(x)
    
    print(f"Input shape:     {x.shape}")
    print(f"Output shape:    {output.shape}")
    
    print("✓ MLP layer test PASSED!")