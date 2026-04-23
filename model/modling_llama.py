"""
LLaMA 3 Model Implementation with Triton Kernels
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from llama3.model.config import LlamaConfig
from modelscope import snapshot_download

from llama3.model.layers.ffn_fused_triton import LlamaMLPTriton
from llama3.model.layers.rms_norm_triton import LlamaRMSNormTriton
from llama3.model.layers.rope_triton import LlamaRotaryEmbeddingTriton
from llama3.model.layers.attention_triton import LlamaAttentionTriton


class LlamaDecoderLayer(nn.Module):
    """
    LLaMA Decoder Layer
    
    结构:
    1. Input -> RMSNorm -> Attention -> +residual
    2. -> RMSNorm -> MLP -> +residual
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        use_triton: bool = True,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Self Attention
        self.self_attn = LlamaAttentionTriton(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            use_triton=use_triton,
        )
        
        # MLP
        self.mlp = LlamaMLPTriton(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_triton=use_triton,
        )
        
        # RMSNorm
        self.input_layernorm = LlamaRMSNormTriton(
            config.hidden_size, 
            eps=config.rms_norm_eps,
            use_triton=use_triton,
        )
        self.post_attention_layernorm = LlamaRMSNormTriton(
            config.hidden_size,
            eps=config.rms_norm_eps,
            use_triton=use_triton,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_embeddings: (cos, sin)
            past_key_value: KV Cache
            use_cache: 是否使用KV Cache
        
        Returns:
            hidden_states, present_key_value
        """
        residual = hidden_states
        
        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class LlamaModel(nn.Module):
    """
    LLaMA Model (Transformer主体)
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        use_triton: bool = True,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx
        )
        
        # Decoder层
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx, use_triton)
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # 最终RMSNorm
        self.norm = LlamaRMSNormTriton(
            config.hidden_size, 
            eps=config.rms_norm_eps,
            use_triton=use_triton,
        )
        
        # RoPE位置编码
        self.rotary_emb = LlamaRotaryEmbeddingTriton(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            theta=config.rope_theta,
            use_triton=use_triton,
        )
        
        # 梯度检查点（节省显存）
        self.gradient_checkpointing = False
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            past_key_values: 列表，每个元素是一个层的KV Cache
            use_cache: 是否返回KV Cache
        
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # 词嵌入
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        # 获取RoPE位置编码
        batch_size, seq_len = input_ids.shape
        if past_key_values is not None and past_key_values[0] is not None:
            # 增量推理：seq_len 应该是 past_length + current_seq_len
            past_length = past_key_values[0][0].size(-2)  # 取 K 的 seq_len
            position_ids = torch.arange(
                past_length, past_length + seq_len,
                dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
        else:
            position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        # 根据 position_ids 取 RoPE，而不是简单 [:seq_len]
        position_embeddings = self.rotary_emb.cos_cached[position_ids], self.rotary_emb.sin_cached[position_ids]
        
        # 处理KV Cache
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # 逐层前向传播
        next_decoder_cache = [] if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            # 前向传播
            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_decoder_cache.append(present_key_value)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        return hidden_states if not use_cache else (hidden_states, next_decoder_cache)


class LlamaForCausalLM(nn.Module):
    """
    LLaMA Model with Language Modeling Head
    
    用于因果语言建模（生成任务）
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        use_triton: bool = True,
    ):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, use_triton=use_triton)
        
        # LM Head
        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # 权重绑定（可选）
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        
        self.use_triton = use_triton
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]，用于计算loss
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: 如果提供了labels
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 获取模型输出
        model_outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,  # 传进去
            use_cache=use_cache,
        )
        
        if use_cache:
            hidden_states, past_key_values = model_outputs
        else:
            hidden_states = model_outputs
            past_key_values = None
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        # 计算loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": logits,
            # "hidden_states": hidden_states,
            "past_key_values": past_key_values,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        简单的文本生成
        
        Args:
            input_ids: [batch, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus sampling
            do_sample: 是否采样
        
        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        batch_size, seq_len = input_ids.shape
        past_key_values = None
        generated = input_ids.clone()  # 维护完整序列用于返回
    
        with torch.no_grad():
            for i in range(max_new_tokens):
                # 只有第一步传完整 prompt，后面只传 1 个 token
                inputs = input_ids if past_key_values is None else next_token.unsqueeze(-1)
            
                outputs = self.forward(
                    input_ids=inputs,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
                logits = outputs["logits"]
                past_key_values = outputs["past_key_values"]
            
                next_token_logits = logits[:, -1, :] / temperature
            
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
                generated = torch.cat([generated, next_token], dim=-1)
            
                # 检查 EOS
                if next_token.item() == getattr(self.config, 'eos_token_id', 2):
                    break
    
        return generated


def create_llama3_2_3b(use_triton: bool = True) -> LlamaForCausalLM:
    """
    创建LLaMA 3.2 3B模型
    
    Args:
        use_triton: 是否使用Triton加速
    
    Returns:
        model: LLaMA 3.2 3B模型
    """
    model_dir = snapshot_download("LLM-Research/Llama-3.2-1B")
    config = LlamaConfig.from_pretrained(model_dir)
    model = LlamaForCausalLM(config, use_triton=use_triton)
    return model


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit()
    
    print("Testing LLaMA 3 Model...")
    
    # 创建模型
    model = create_llama3_2_3b(use_triton=True).cuda().half()
    
    # 测试输入
    batch_size = 1
    seq_len = 32
    input_ids = torch.randint(0, 128256, (batch_size, seq_len), device="cuda")
    
    # 前向传播
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
    
    
    print(f"Input shape:    {input_ids.shape}")
    print(f"Logits shape:   {outputs['logits'].shape}")
    print(f"Loss:           {outputs['loss']}")
    
    print("\\n✓ LLaMA 3 Model test PASSED!")