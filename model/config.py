"""
LLaMA 3 模型配置类
模仿transformers风格的配置
"""


class LlamaConfig:
    """
    LLaMA 3.2 3B 模型配置
    """
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 3072,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,  # GQA
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        pad_token_id: int = 0,
        bos_token_id: int = 128000,
        eos_token_id: int = 128001,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        
        # 计算head_dim
        self.head_dim = hidden_size // num_attention_heads
        
        # 额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def llama3_2_3b(cls):
        """LLaMA 3.2 3B 官方配置"""
        return cls(
            vocab_size=128256,
            hidden_size=3072,
            intermediate_size=8192,
            num_hidden_layers=28,
            num_attention_heads=24,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
        )
    
    def to_dict(self):
        """转换为字典"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
        }