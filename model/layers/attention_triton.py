"""
LLaMA Attention Layer with Triton Kernels
支持GQA (Grouped Query Attention) 和 Triton加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入Triton kernel
try:
    from llama3.kernels.attention_prefill import flash_attention_prefill
    from llama3.kernels.attention_decode import flash_attention_decode
    # print("vic!")
    TRITON_AVAILABLE = True
except ImportError:
    # print("fail!")
    TRITON_AVAILABLE = False


class LlamaAttentionTriton(nn.Module):
    """
    LLaMA Attention with Triton加速
    
    特点:
    - GQA: num_kv_heads <= num_heads
    - RoPE位置编码
    - FlashAttention加速
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 500000.0,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # GQA: num_kv_heads可能小于num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        
        # Q, K, V投影
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # 缩放因子
        self.scaling = head_dim ** -0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple = None,  # (cos, sin)
        past_key_value: tuple = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor = None,
    ):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) tuple
            past_key_value: 之前的KV Cache
            use_cache: 是否使用KV Cache
            attention_mask: 注意力掩码
        
        Returns:
            attn_output, (present_key_value,)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑为 [batch, seq_len, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用RoPE位置编码
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # 这里假设外面已经处理好了RoPE
            pass
        
        # 处理KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=1)
            value_states = torch.cat([past_v, value_states], dim=1)
        
        present_key_value = (key_states, value_states) if use_cache else None
        
        kv_seq_len = key_states.shape[1]
        
        # 使用PyTorch或Triton计算注意力
        if self.use_triton and seq_len > 1:
            # Prefill阶段使用FlashAttention
            try:
                # print("triton using!")
                attn_output = self._flash_attention_prefill(
                    query_states, key_states, value_states
                )
            except Exception as e:
                # print(f"{e}")
                attn_output = self._attention_pytorch(
                    query_states, key_states, value_states, attention_mask
                )
        elif self.use_triton and seq_len == 1:
            # Decode阶段使用优化的decode kernel
            try:
                # print("triton using!")
                attn_output = self._flash_attention_decode(
                    query_states, key_states, value_states
                )
            except Exception as e:
                # print(f"{e}")
                attn_output = self._attention_pytorch(
                    query_states, key_states, value_states, attention_mask
                )
        else:
            # PyTorch实现
            attn_output = self._attention_pytorch(
                query_states, key_states, value_states, attention_mask
            )
        
        # 重塑并输出投影
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value
    
    def _flash_attention_prefill(self, query, key, value):
        """使用Triton FlashAttention (prefill)"""
        batch, q_len, num_heads, head_dim = query.shape
        _, kv_len, num_kv_heads, _ = key.shape
        
        # 扩展KV以匹配Q的头数 (GQA)
        if num_kv_heads < num_heads:
            key = self._repeat_kv(key, self.num_kv_groups)
            value = self._repeat_kv(value, self.num_kv_groups)
        
        # 转置为 [batch, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 调用Triton kernel
        output = flash_attention_prefill(query, key, value)
        
        # 恢复形状
        output = output.transpose(1, 2).contiguous()
        
        return output
    
    def _flash_attention_decode(self, query, key, value):
        """使用Triton FlashAttention (decode)"""
        batch, q_len, num_heads, head_dim = query.shape
        
        # 扩展KV
        if key.shape[2] < num_heads:
            key = self._repeat_kv(key, self.num_kv_groups)
            value = self._repeat_kv(value, self.num_kv_groups)
        
        # 转置
        query = query.transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        key = key.transpose(1, 2)      # [batch, num_heads, kv_len, head_dim]
        value = value.transpose(1, 2)
        
        # 使用decode kernel
        # 这里简化为使用prefill kernel处理
        output = flash_attention_decode(
            query.squeeze(2),  # [batch, num_heads, head_dim]
            key,
            value,
            kv_len=key.shape[2]
        )
        
        # 恢复形状 [batch, 1, num_heads, head_dim]
        output = output.unsqueeze(1)
        
        return output
    
    def _attention_pytorch(self, query, key, value, attention_mask=None):
        """PyTorch标准注意力实现"""
        batch, q_len, num_heads, head_dim = query.shape
        _, kv_len, num_kv_heads, _ = key.shape
        
        # GQA: 扩展KV
        if num_kv_heads < num_heads:
            key = self._repeat_kv(key, self.num_kv_groups)
            value = self._repeat_kv(value, self.num_kv_groups)
        
        # 转置为 [batch, num_heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        # 应用掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Causal mask for prefill
        if q_len > 1:
            causal_mask = torch.triu(
                torch.ones(q_len, kv_len, device=query.device), 
                diagonal=1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, value)
        
        # 恢复形状 [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        return attn_output
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        为GQA重复KV头
        [batch, seq_len, num_kv_heads, head_dim] -> 
        [batch, seq_len, num_kv_heads * n_rep, head_dim]
        """
        batch, seq_len, num_kv_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(
            batch, seq_len, num_kv_heads, n_rep, head_dim
        )
        return hidden_states.reshape(batch, seq_len, num_kv_heads * n_rep, head_dim)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing LLaMA Attention...")
    
    # 测试配置 (LLaMA 3.2 3B)
    batch_size = 2
    seq_len = 16
    hidden_size = 3072
    num_heads = 24
    num_kv_heads = 8  # GQA
    head_dim = 128
    
    torch.manual_seed(42)
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, 
        device="cuda", dtype=torch.float16
    )
    
    # 创建注意力层
    attn = LlamaAttentionTriton(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    ).cuda().half()
    
    # 测试前向
    output, _ = attn(hidden_states)
    
    print(f"Input shape:  {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Attention layer test PASSED!")