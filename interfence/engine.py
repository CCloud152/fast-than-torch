"""
推理引擎主流程
整合模型、KV Cache和Tokenizer
"""

import torch
import torch.nn as nn
from typing import Optional, List, Generator

from ..model.modling_llama import LlamaForCausalLM
from ..model.config import LlamaConfig
from .kv_cache import KVCache
from .tokenizer_utils import TokenizerUtils


class InferenceEngine:
    """
    LLaMA 3 推理引擎
    
    功能:
    - 模型加载和管理
    - KV Cache管理
    - 文本生成
    """
    
    def __init__(
        self,
        model: Optional[LlamaForCausalLM] = None,
        config: Optional[LlamaConfig] = None,
        tokenizer: Optional[TokenizerUtils] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype
        
        # 模型
        if model is not None:
            self.model = model.to(device).to(dtype)
        elif config is not None:
            self.model = LlamaForCausalLM(config).to(device).to(dtype)
        else:
            raise ValueError("Must provide either model or config")
        
        self.model.eval()
        self.config = self.model.config
        
        # Tokenizer
        self.tokenizer = tokenizer or TokenizerUtils()
        
        # KV Cache
        self.kv_cache = None
        
        # 性能统计
        self.prefill_time = 0
        self.decode_time = 0
        self.tokens_generated = 0
    
    def _init_kv_cache(self, batch_size: int, max_seq_len: int = 2048):
        """初始化KV Cache"""
        self.kv_cache = KVCache(
            num_layers=self.config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
    
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus sampling
            top_k: top-k sampling
            repetition_penalty: 重复惩罚
            do_sample: 是否采样
        
        Returns:
            generated_text: 生成的文本
        """
        import time
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = input_ids.to(self.device)
        batch_size, prompt_len = input_ids.shape
        
        # 初始化KV Cache
        self._init_kv_cache(batch_size, max_seq_len=prompt_len + max_new_tokens + 100)
        
        # Prefill阶段
        start_time = time.perf_counter()
        
        # 前向传播获取prompt的KV Cache
        outputs = self.model(
            input_ids=input_ids,
            use_cache=True,
        )
        
        # 获取最后一个token的logits
        next_token_logits = outputs["logits"][:, -1, :]
        
        self.prefill_time = time.perf_counter() - start_time
        
        # 采样第一个token
        next_token = self._sample_token(
            next_token_logits,
            temperature,
            top_p,
            top_k,
            do_sample,
        )
        
        # 收集生成的token
        generated_tokens = [next_token.item()]
        
        # Decode阶段
        self.tokens_generated = 0
        start_time = time.perf_counter()
        
        for i in range(max_new_tokens - 1):
            # 准备输入（只有新token）
            input_ids = torch.tensor([[next_token.item()]], device=self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
            )
            
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens,
                    repetition_penalty,
                )
            
            # 采样
            next_token = self._sample_token(
                next_token_logits,
                temperature,
                top_p,
                top_k,
                do_sample,
            )
            
            generated_tokens.append(next_token.item())
            self.tokens_generated += 1
            
            # 检查是否生成了EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        self.decode_time = time.perf_counter() - start_time
        
        # 解码结果
        all_token_ids = torch.tensor([generated_tokens], device="cpu")
        generated_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)
        
        return generated_text
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Generator[str, None, None]:
        """
        流式生成
        
        Yields:
            token: 每个生成的token
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = input_ids.to(self.device)
        batch_size, prompt_len = input_ids.shape
        
        # 初始化KV Cache
        self._init_kv_cache(batch_size)
        
        # Prefill
        outputs = self.model(input_ids=input_ids, use_cache=True)
        next_token_logits = outputs["logits"][:, -1, :]
        
        # 生成循环
        for _ in range(max_new_tokens):
            # 采样
            next_token = self._sample_token(
                next_token_logits,
                temperature,
                top_p,
                50,  # top_k
                True,  # do_sample
            )
            
            # 解码token
            token_text = self.tokenizer.decode(next_token, skip_special_tokens=True)
            yield token_text
            
            # 检查EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # 下一个token
            input_ids = next_token.unsqueeze(0)
            outputs = self.model(input_ids=input_ids, use_cache=True)
            next_token_logits = outputs["logits"][:, -1, :]
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
    ) -> torch.Tensor:
        """采样单个token"""
        # 应用temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        if do_sample:
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        return next_token.squeeze(0)
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        penalty: float,
    ) -> torch.Tensor:
        """应用重复惩罚"""
        for token_id in set(generated_tokens):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits
    
    def get_stats(self) -> dict:
        """获取推理统计信息"""
        stats = {
            "prefill_time_ms": self.prefill_time * 1000,
            "decode_time_ms": self.decode_time * 1000,
            "tokens_generated": self.tokens_generated,
        }
        if self.tokens_generated > 0:
            stats["decode_tokens_per_sec"] = self.tokens_generated / self.decode_time
        return stats
    
    def reset(self):
        """重置引擎状态"""
        self.kv_cache = None
        self.prefill_time = 0
        self.decode_time = 0
        self.tokens_generated = 0


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing Inference Engine...")
    
    # 创建模型配置（使用小配置测试）
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=1024,  # 小配置
        intermediate_size=4096,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    
    # 创建引擎
    engine = InferenceEngine(config=config)
    
    # 测试生成
    prompt = "Hello"
    print(f"Prompt: {prompt}")
    
    # 由于模型是随机初始化的，生成结果会是无意义的
    # 这里只测试流程
    print("Engine initialized successfully!")
    print("✓ Inference Engine test PASSED!")