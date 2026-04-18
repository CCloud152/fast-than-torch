```
triton-inference-lab/
├── README.md                 # 项目简报
├── requirements.txt          # torch, triton, transformers
│
├── model/                    # 模型定义（PyTorch + Triton混合）
│   ├── __init__.py
│   ├── config.py             # 模型配置类（模仿transformers风格）
│   ├── modeling_llama.py     # 简化版Llama实现（RMSNorm, RoPE, Attention, FFN）
│   └── layers/
│       ├── __init__.py
│       ├── rms_norm_triton.py    # Triton融合算子：RMSNorm + Residual
│       ├── rope_triton.py        # Triton融合算子：RoPE计算
│       ├── ffn_fised_triton.py   # Triton融合算子：MLP计算 
│       └── attention_triton.py   # Triton融合算子：简化FlashAttention
│
├── kernels/                  # Triton算子库（核心实验内容）
│   ├── __init__.py
│   ├── rms_norm_fused.py   # 融合RMSNorm：norm + residual + activation
│   ├── rope_fused.py       # 融合RoPE：sin/cos计算 + 旋转
│   ├── ffn_fused.py        # 融合FFN：matmul计算 + SiLU/SwiGLU
│   ├── attention_decode.py # 简化版FlashAttention for decode
│   └── attention_prefill.py # 简化版FlashAttention for prefill
│
├── inference/               # 推理引擎流程
│   ├── __init__.py
│   ├── engine.py            # 推理主流程（KV Cache管理）
│   ├── kv_cache.py          # KV Cache实现（page方式或连续方式）
│   └── tokenizer_utils.py   # 文本处理
│
└── tests/                   # 验证代码
    ├── test_e2e_inference.py       # 端到端推理测试
    └── benchmark_speed.py          # 速度对比基准测试
```
用 Triton 算子替换 PyTorch 算子，超越原生 torch 性能跑 LLaMA 3.2 3B

**针对 LLaMA 结构做专用优化**  
- 单 kernel 融合（把多层计算揉进一个核）  
- 显存 0 拷贝  
- 专门为 LLM 优化的访存模式  
- 更低的 launch overhead  
- 更高的 SM 利用率