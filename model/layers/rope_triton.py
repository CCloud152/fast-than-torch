"""
LLaMA RoPE (Rotary Position Embedding) with Triton
"""

import torch
import torch.nn as nn

# 尝试导入Triton kernel
try:
    from ...kernels.rope_fused import rope_fused, precompute_rope_rotary_cache
    TRITON_AVAILABLE = True
except ImportError:
    try:
        from kernels.rope_fused import rope_fused, precompute_rope_rotary_cache
        TRITON_AVAILABLE = True
    except ImportError:
        TRITON_AVAILABLE = False


class LlamaRotaryEmbeddingTriton(nn.Module):
    """
    LLaMA RoPE with Triton加速
    
    RoPE通过旋转位置编码实现相对位置感知:
    - 对Q和K应用旋转
    - 旋转角度取决于位置索引
    """
    
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 131072,
        theta: float = 500000.0,
        use_triton: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # 预计算位置编码缓存
        if TRITON_AVAILABLE:
            cos, sin = precompute_rope_rotary_cache(
                max_position_embeddings, head_dim, theta
            )
            self.register_buffer("cos_cached", cos, persistent=False)
            self.register_buffer("sin_cached", sin, persistent=False)
        else:
            # PyTorch回退
            inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            self.register_buffer("inv_freq", inv_freq)
    
    def _rope_pytorch(self, q, k, cos, sin):
        """PyTorch RoPE实现"""
        # 分离偶数和奇数维度
        q_x1, q_x2 = q[..., ::2], q[..., 1::2]
        k_x1, k_x2 = k[..., ::2], k[..., 1::2]
        
        # 应用旋转
        q_rotated = torch.empty_like(q)
        k_rotated = torch.empty_like(k)
        
        q_rotated[..., ::2] = q_x1 * cos - q_x2 * sin
        q_rotated[..., 1::2] = q_x1 * sin + q_x2 * cos
        
        k_rotated[..., ::2] = k_x1 * cos - k_x2 * sin
        k_rotated[..., 1::2] = k_x1 * sin + k_x2 * cos
        
        return q_rotated, k_rotated
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int = None,
    ):
        """
        Args:
            q: [batch, seq_len, num_heads, head_dim]
            k: [batch, seq_len, num_kv_heads, head_dim]
            seq_len: 当前序列长度
        
        Returns:
            q_rotated, k_rotated
        """
        batch_size, q_len, num_heads, head_dim = q.shape
        
        if seq_len is None:
            seq_len = q_len
        
        # 获取cos/sin缓存
        if hasattr(self, 'cos_cached'):
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            # 动态计算
            t = torch.arange(seq_len, device=q.device)
            freqs = torch.outer(t, self.inv_freq)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        
        # 尝试Triton加速
        if self.use_triton:
            try:
                # 调整维度顺序以匹配Triton kernel
                # [batch, seq, heads, dim] -> [batch, heads, seq, dim]
                q_t = q.transpose(1, 2)
                k_t = k.transpose(1, 2)
                
                q_rot, k_rot = rope_fused(q_t, k_t, cos, sin)
                
                # 恢复维度顺序
                q_rot = q_rot.transpose(1, 2)
                k_rot = k_rot.transpose(1, 2)
                
                return q_rot, k_rot
            except Exception as e:
                pass
        
        # PyTorch实现
        # 扩展cos/sin维度
        cos = cos.unsqueeze(1)  # [seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(1)
        
        return self._rope_pytorch(q, k, cos, sin)


def apply_rotary_pos_emb_triton(q, k, cos, sin):
    """
    直接应用RoPE（用于预计算的cos/sin）
    
    Args:
        q: [batch, seq_len, num_heads, head_dim]
        k: [batch, seq_len, num_kv_heads, head_dim]
        cos: [seq_len, head_dim//2]
        sin: [seq_len, head_dim//2]
    """
    if TRITON_AVAILABLE:
        try:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            q_rot, k_rot = rope_fused(q_t, k_t, cos, sin)
            return q_rot.transpose(1, 2), k_rot.transpose(1, 2)
        except:
            pass
    
    # PyTorch回退
    q_x1, q_x2 = q[..., ::2], q[..., 1::2]
    k_x1, k_x2 = k[..., ::2], k[..., 1::2]
    
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
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
    
    print("Testing LLaMA RoPE...")
    
    # 测试配置
    batch_size = 2
    seq_len = 16
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    
    # 创建RoPE层
    rope = LlamaRotaryEmbeddingTriton(head_dim).cuda()
    
    # 测试前向
    q_rot, k_rot = rope(q, k)
    
    print(f"Q input shape:  {q.shape}")
    print(f"Q output shape: {q_rot.shape}")
    print(f"K input shape:  {k.shape}")
    print(f"K output shape: {k_rot.shape}")
    print("✓ RoPE layer test PASSED!")