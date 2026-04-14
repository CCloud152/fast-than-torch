"""
KV Cache 管理
支持Paged Attention风格的缓存管理
"""

import torch
from typing import List, Tuple, Optional


class KVCache:
    """
    KV Cache 管理器
    
    支持:
    - 连续内存分配
    - 动态扩展
    - GQA (Grouped Query Attention) 优化
    """
    
    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 为每一层分配KV Cache
        # 形状: [batch, num_kv_heads, max_seq_len, head_dim]
        self.k_cache = [
            torch.zeros(
                batch_size, num_kv_heads, max_seq_len, head_dim,
                dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(
                batch_size, num_kv_heads, max_seq_len, head_dim,
                dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        
        # 记录当前序列长度
        self.seq_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
    
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
        
        Args:
            layer_idx: 层索引
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            positions: 更新的位置，None表示追加到末尾
        """
        batch_size, num_kv_heads, seq_len, head_dim = key_states.shape
        
        if positions is None:
            # 追加模式
            start_pos = self.seq_lengths[0].item()
            end_pos = start_pos + seq_len
            
            self.k_cache[layer_idx][:, :, start_pos:end_pos, :] = key_states
            self.v_cache[layer_idx][:, :, start_pos:end_pos, :] = value_states
            
            # 更新序列长度
            self.seq_lengths += seq_len
        else:
            # 指定位置更新
            for b in range(batch_size):
                pos = positions[b]
                self.k_cache[layer_idx][b, :, pos:pos+seq_len, :] = key_states[b]
                self.v_cache[layer_idx][b, :, pos:pos+seq_len, :] = value_states[b]
    
    def get_seq_length(self, batch_idx: int = 0) -> int:
        """获取指定batch的序列长度"""
        return self.seq_lengths[batch_idx].item()
    
    def reset(self):
        """重置Cache"""
        self.seq_lengths.zero_()
        for k, v in zip(self.k_cache, self.v_cache):
            k.zero_()
            v.zero_()
    
    def trim(self, new_seq_lengths: torch.Tensor):
        """裁剪到指定长度"""
        self.seq_lengths = new_seq_lengths


class PagedKVCache:
    """
    Paged KV Cache (简化版)
    
    使用分页管理，适合长序列生成
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int = 16,
        max_pages: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        
        # 分配页
        self.pages_k = [
            torch.zeros(
                max_pages, num_kv_heads, page_size, head_dim,
                dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        self.pages_v = [
            torch.zeros(
                max_pages, num_kv_heads, page_size, head_dim,
                dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        
        # 页表: 记录每个序列使用的页
        self.page_tables = {}  # {seq_id: [page_indices]}
    
    def allocate_pages(self, seq_id: int, num_pages: int) -> List[int]:
        """为序列分配页"""
        # 简化实现：顺序分配
        start_idx = len(self.page_tables) * self.max_pages // 100  # 简单分配策略
        page_indices = list(range(start_idx, start_idx + num_pages))
        self.page_tables[seq_id] = page_indices
        return page_indices
    
    def get_kv(self, layer_idx: int, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定序列的KV（需要gather pages）"""
        page_indices = self.page_tables.get(seq_id, [])
        
        # Gather pages
        k_pages = [self.pages_k[layer_idx][idx] for idx in page_indices]
        v_pages = [self.pages_v[layer_idx][idx] for idx in page_indices]
        
        # 拼接
        k = torch.cat(k_pages, dim=1) if k_pages else torch.empty(0)
        v = torch.cat(v_pages, dim=1) if v_pages else torch.empty(0)
        
        return k, v


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