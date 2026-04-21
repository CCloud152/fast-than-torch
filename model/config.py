"""
config.py
自定义 LlamaConfig，封装 ModelScope 下载、config.json 读取、字段映射与补齐。
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


@dataclass
class LlamaConfig:
    """
    与modling_llama完全对齐的 LLaMA 配置类。
    支持从本地路径或 ModelScope ID 一键加载。
    """

    # ========== 核心架构（模型构建必需） ==========
    vocab_size: int = 128256
    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_hidden_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 131072

    # ========== 归一化 & 激活 ==========
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"

    # ========== RoPE（模型构建必需） ==========
    rope_theta: float = 500000.0

    # ========== Token ID（Embedding 必需） ==========
    bos_token_id: int = 128000
    eos_token_id: int = 128001
    pad_token_id: int = 128001          # 真实 config 为 null，这里默认对齐 eos

    # ========== 开关/其他（模型/训练逻辑使用） ==========
    use_cache: bool = True
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    initializer_range: float = 0.02

    # ========== Llama 3 长上下文扩展参数（可选） ==========
    rope_scaling: Optional[Dict[str, Any]] = field(default=None)

    # ========== 内部记录（非模型构建必需，但加载权重时有用） ==========
    torch_dtype: str = "bfloat16"
    model_type: str = "llama"

    # --------------------------------------------------------------------- #
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "LlamaConfig":
        """
        统一入口：支持本地路径 或 ModelScope 模型 ID。

        用法:
            config = LlamaConfig.from_pretrained("LLM-Research/Llama-3.2-3B")
            config = LlamaConfig.from_pretrained("/abs/path/to/local/model")
        """
        # 1. 判断是本地路径还是 ModelScope ID
        if os.path.isdir(model_name_or_path):
            # 本地路径
            model_dir = model_name_or_path
        else:
            # 视为 ModelScope ID，自动下载
            try:
                from modelscope import snapshot_download
            except ImportError:
                raise ImportError(
                    "使用 ModelScope ID 需要安装 modelscope: pip install modelscope"
                )
            model_dir = snapshot_download(model_name_or_path)

        # 2. 读取 config.json
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"在 {model_dir} 中未找到 config.json，请确认模型已正确下载。"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        # 3. 字段映射 + 补齐
        return cls._from_dict(raw_cfg)

    @classmethod
    def _from_dict(cls, raw: Dict[str, Any]) -> "LlamaConfig":
        """
        从原始 config.json 字典构建，处理所有字段映射与缺失值。
        """
        kwargs: Dict[str, Any] = {}

        # --- 3.1 直接复制：两边字段名完全一致的 ---
        direct_copy = [
            "vocab_size", "hidden_size", "intermediate_size",
            "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "max_position_embeddings",
            "rms_norm_eps", "use_cache", "pad_token_id",
            "bos_token_id", "eos_token_id", "tie_word_embeddings",
            "attention_bias", "attention_dropout", "mlp_bias",
            "hidden_act", "initializer_range", "torch_dtype", "model_type",
        ]
        for k in direct_copy:
            if k in raw:
                kwargs[k] = raw[k]

        # --- 3.2 处理 rope_theta（兼容两种格式）---
        if "rope_theta" in raw:
            kwargs["rope_theta"] = raw["rope_theta"]
        elif "rope_parameters" in raw and isinstance(raw["rope_parameters"], dict):
            kwargs["rope_theta"] = raw["rope_parameters"].get("rope_theta", 500000.0)
        else:
            kwargs["rope_theta"] = 500000.0

        # --- 3.3 处理 rope_scaling（Llama 3 的 128K 上下文配置）---
        if "rope_scaling" in raw:
            kwargs["rope_scaling"] = raw["rope_scaling"]
        elif "rope_parameters" in raw and isinstance(raw["rope_parameters"], dict):
            rope_params = dict(raw["rope_parameters"])
            rope_params.pop("rope_theta", None)
            if rope_params:
                kwargs["rope_scaling"] = rope_params

        # --- 3.4 补齐 pad_token_id（真实 config 为 null，必须给整数值）---
        if kwargs.get("pad_token_id") is None:
            kwargs["pad_token_id"] = kwargs.get("eos_token_id", 128001)

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """导出字典，方便打印调试"""
        return self.__dict__.copy()

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for k, v in sorted(self.to_dict().items()):
            lines.append(f"  {k}={v},")
        lines.append(")")
        return "\n".join(lines)


# ==================== 测试入口 ====================
if __name__ == "__main__":
    # 示例 1：从 ModelScope ID 加载（自动下载）
    print("=" * 50)
    print("方式 1：从 ModelScope ID 加载")
    print("=" * 50)
    config = LlamaConfig.from_pretrained("LLM-Research/Llama-3.2-3B")
    print(config)

    # 示例 2：从本地已下载路径加载
    # config_local = LlamaConfig.from_pretrained(
    #     "/home/karawink/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-3B"
    # )