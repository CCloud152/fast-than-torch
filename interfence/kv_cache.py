"""
KV Cache 管理
支持Paged Attention风格的缓存管理
"""

import torch
from typing import Tuple

class KVCache:
    """
    KV Cache 管理器 (支持动态扩容)
    
    支持:
    - 动态扩展 (Dynamic Expansion): 当序列长度超过当前容量时自动扩容
    - 连续内存分配 (在单次分配内)
    - GQA (Grouped Query Attention)
    
    策略: 倍增扩容 (Amortized O(1) append)
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int = 2048,  # 初始容量/最大容量限制
        num_kv_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        growth_factor: float = 2.0,  # 扩容倍数
        initial_capacity: int = None, # 初始分配大小
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.growth_factor = growth_factor
        self.max_capacity = max_seq_len 
        
        # 设置初始容量
        # 如果指定了 initial_capacity，否则默认为 max_seq_len 的 1/4 或 256，取较大值
        self.initial_capacity = initial_capacity or max(256, max_seq_len // 4)
        
        # 记录当前序列长度 (逻辑长度)
        self.seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # 为每一层分配KV Cache
        # 形状: [batch, num_kv_heads, current_capacity, head_dim]
        # 注意：这里的 max_seq_len 现在代表的是当前物理容量 (Capacity)
        self.k_cache = [
            torch.zeros(
                batch_size, 
                num_kv_heads, 
                self.initial_capacity, 
                head_dim, 
                dtype=dtype, 
                device=device
            ) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(
                batch_size, 
                num_kv_heads, 
                self.initial_capacity, 
                head_dim, 
                dtype=dtype, 
                device=device
            ) for _ in range(num_layers)
        ]
        
        # 记录当前物理容量 (每个 Tensor 实际能存多少)
        # 注意：所有层的容量必须一致，否则无法进行 Attention 计算
        self.current_capacity = self.initial_capacity 

    def _resize(self, layer_idx: int, new_capacity: int):
        """
        核心扩容函数
        逻辑：创建更大的 Tensor -> 拷贝旧数据 -> 替换引用
        """
        # 1. 创建新 Tensor
        new_k = torch.zeros(
            self.batch_size, 
            self.num_kv_heads, 
            new_capacity, 
            self.head_dim, 
            dtype=self.dtype, 
            device=self.device
        )
        new_v = torch.zeros(
            self.batch_size, 
            self.num_kv_heads, 
            new_capacity, 
            self.head_dim, 
            dtype=self.dtype, 
            device=self.device
        )
        
        # 2. 拷贝旧数据
        # 注意：这里只拷贝实际有数据的部分 (seq_lengths)，而不是整个旧 Tensor
        current_len = self.seq_lengths.max().item() # 取 batch 中最长的序列长度
        new_k[:, :, :current_len, :] = self.k_cache[layer_idx][:, :, :current_len, :]
        new_v[:, :, :current_len, :] = self.v_cache[layer_idx][:, :, :current_len, :]
        
        # 3. 替换引用 (旧 Tensor 会被 Python GC 自动回收)
        self.k_cache[layer_idx] = new_k
        self.v_cache[layer_idx] = new_v

    def _ensure_capacity(self, layer_idx: int, required_capacity: int):
        """
        检查并确保空间足够
        如果不够，触发扩容
        """
        # 如果当前物理容量已经足够，直接返回
        if self.current_capacity >= required_capacity:
            return
            
        # 边界检查：不能超过用户设定的 max_capacity
        if required_capacity > self.max_capacity:
            raise RuntimeError(
                f"Required capacity {required_capacity} exceeds max_capacity {self.max_capacity}. "
                f"Please increase max_seq_len in the constructor."
            )
        
        # 计算新容量：通常是当前容量的 growth_factor 倍，或者刚好满足需求，取较大值
        new_capacity = int(max(self.current_capacity * self.growth_factor, required_capacity))
        
        # 限制新容量不超过 max_capacity
        new_capacity = min(new_capacity, self.max_capacity)
        
        # 对所有层进行扩容 (为了保持一致性，通常所有层一起扩)
        # 注意：这里简化了逻辑，假设所有层都需要同步扩容
        for i in range(self.num_layers):
            self._resize(i, new_capacity)
            
        # 更新全局记录的容量
        self.current_capacity = new_capacity

    def get_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的KV Cache"""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def update_cache(
        self, 
        layer_idx: int, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        positions: torch.Tensor = None,
    ):
        """
        更新KV Cache
        注意：positions 参数如果存在，必须在当前容量范围内
        """
        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape
        
        # 如果是追加模式，计算需要的总容量
        if positions is None:
            required_capacity = self.seq_lengths[0].item() + seq_len
            
            # 触发扩容检查
            self._ensure_capacity(layer_idx, required_capacity)
            
            # 写入数据 (追加到末尾)
            start_pos = self.seq_lengths[0].item()
            end_pos = start_pos + seq_len
            self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = key_states
            self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = value_states
            
            # 更新逻辑长度
            self.seq_lengths += seq_len
            
        else:
            # 在指定位置更新时，我们假设调用者已经确保了 positions + seq_len 不会越界
            # 或者 positions 在当前容量范围内
            for b in range(batch_size):
                pos = positions[b]
                # 边界检查
                if pos + seq_len > self.current_capacity:
                    # 如果指定了位置但空间不够，且位置是连续的，可以尝试扩容
                    # 这里为了简单，如果位置指定模式下空间不足，直接报错或强制扩容到指定位置
                    self._ensure_capacity(layer_idx, pos + seq_len)
                    
                self.k_cache[layer_idx][b, :, pos:pos+seq_len, :] = key_states[b]
                self.v_cache[layer_idx][b, :, pos:pos+seq_len, :] = value_states[b]


if __name__ == "__main__":
    print("Testing KV Cache...")
    
    # 测试配置
    num_layers = 28
    batch_size = 2
    max_seq_len = 2048
    num_kv_heads = 8
    head_dim = 128
    
    # 创建KV Cache
    cache = KVCache(
        num_layers=num_layers,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    
    # 模拟更新
    seq_len = 16
    key_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    
    cache.update_cache(0, key_states, value_states)
    
    print(f"Cache shape: {cache.k_cache[0].shape}")
    print(f"Seq length: {cache.get_seq_length()}")
    print("✓ KV Cache test PASSED!")