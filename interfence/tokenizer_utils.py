"""
Tokenizer 工具类
处理文本编码和解码
"""

import torch
from typing import List, Union


class TokenizerUtils:
    """
    简化版Tokenizer工具
    
    实际使用时应该加载HuggingFace的AutoTokenizer
    这里提供一个包装接口
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.bos_token_id = 128000  # LLaMA 3 BOS
        self.eos_token_id = 128001  # LLaMA 3 EOS
        self.pad_token_id = 128001
    
    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: int = None,
    ) -> torch.Tensor:
        """
        编码文本为token IDs
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊token
            max_length: 最大长度
        
        Returns:
            token_ids: [batch, seq_len]
        """
        if self.tokenizer is not None:
            # 使用实际的tokenizer
            encoded = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=True if max_length else False,
                padding=True,
                return_tensors="pt",
            )
            return encoded["input_ids"]
        
        # 简化实现：模拟编码
        # 实际使用时应该加载真实tokenizer
        if isinstance(text, str):
            text = [text]
        
        # 这里使用随机token作为示例
        # 实际应该是 tokenizer.encode(text)
        seq_lengths = [len(t.split()) for t in text]
        max_len = max(seq_lengths) if max_length is None else min(max(seq_lengths), max_length)
        
        token_ids = []
        for t in text:
            # 简单分词：按空格分割
            tokens = t.split()[:max_len]
            ids = [hash(w) % 128000 + 100 for w in tokens]
            
            if add_special_tokens:
                ids = [self.bos_token_id] + ids
            
            # 填充
            while len(ids) < max_len + (1 if add_special_tokens else 0):
                ids.append(self.pad_token_id)
            
            token_ids.append(ids)
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        解码token IDs为文本
        
        Args:
            token_ids: [batch, seq_len] 或 [seq_len]
            skip_special_tokens: 是否跳过特殊token
        
        Returns:
            text: 解码后的文本
        """
        if self.tokenizer is not None:
            return self.tokenizer.decode(
                token_ids.tolist(),
                skip_special_tokens=skip_special_tokens,
            )
        
        # 简化实现
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        texts = []
        for ids in token_ids.tolist():
            tokens = []
            for id in ids:
                if skip_special_tokens and id in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                    continue
                # 模拟解码
                tokens.append(f"<token_{id}>")
            texts.append(" ".join(tokens))
        
        return texts[0] if len(texts) == 1 else texts
    
    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = None,
    ) -> torch.Tensor:
        """批量编码"""
        return self.encode(
            texts,
            add_special_tokens=True,
            max_length=max_length,
        )
    
    @staticmethod
    def load_from_pretrained(model_name: str = "meta-llama/Llama-3.2-3B"):
        """
        从HuggingFace加载真实tokenizer
        
        Args:
            model_name: 模型名称
        
        Returns:
            TokenizerUtils实例
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            return TokenizerUtils(tokenizer)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            return TokenizerUtils()


if __name__ == "__main__":
    print("Testing Tokenizer Utils...")
    
    # 创建tokenizer
    tokenizer = TokenizerUtils()
    
    # 测试编码
    texts = ["Hello world", "This is a test"]
    token_ids = tokenizer.batch_encode(texts)
    
    print(f"Input texts: {texts}")
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Token IDs: {token_ids}")
    
    # 测试解码
    decoded = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded}")
    
    print("✓ Tokenizer Utils test PASSED!")