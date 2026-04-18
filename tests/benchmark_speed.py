"""
速度基准测试
对比PyTorch和Triton实现的性能
"""

import torch
import time
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoConfig

from model.modling_llama import LlamaForCausalLM


def benchmark_rmsnorm():
    """测试RMSNorm性能"""
    print("\n=== Benchmark: RMSNorm ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from model.layers.rms_norm_triton import LlamaRMSNormTriton
    
    batch_size, seq_len, hidden_size = 8, 512, 3072
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    
    # PyTorch版本
    norm_torch = LlamaRMSNormTriton(hidden_size, use_triton=False).cuda()
    
    # Triton版本
    norm_triton = LlamaRMSNormTriton(hidden_size, use_triton=True).cuda()
    
    # Warmup
    for _ in range(10):
        _ = norm_torch(x)
        _ = norm_triton(x)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    n_iters = 100
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = norm_torch(x)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000
    
    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = norm_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"PyTorch RMSNorm: {pytorch_time:.3f} ms")
    print(f"Triton RMSNorm:  {triton_time:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")


def benchmark_ffn():
    """测试FFN性能"""
    print("\n=== Benchmark: FFN (SwiGLU) ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from model.modling_llama import LlamaMLPTriton
    
    batch_size, seq_len = 8, 512
    hidden_size = 3072
    intermediate_size = 8192
    
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    
    # PyTorch版本
    mlp_torch = LlamaMLPTriton(hidden_size, intermediate_size, use_triton=False).cuda().half()
    
    # Triton版本
    mlp_triton = LlamaMLPTriton(hidden_size, intermediate_size, use_triton=True).cuda().half()
    
    # Warmup
    for _ in range(10):
        _ = mlp_torch(x)
        _ = mlp_triton(x)
    torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 50
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = mlp_torch(x)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000
    
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = mlp_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"PyTorch FFN: {pytorch_time:.3f} ms")
    print(f"Triton FFN:  {triton_time:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")


def benchmark_attention():
    """测试Attention性能"""
    print("\n=== Benchmark: Attention ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from triton_infer.llama3.model.layers.attention_triton import LlamaAttentionTriton
    
    batch_size, seq_len = 2, 1024
    hidden_size = 3072
    num_heads = 24
    num_kv_heads = 8
    head_dim = 128
    
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    
    # PyTorch版本
    attn_torch = LlamaAttentionTriton(
        hidden_size, num_heads, num_kv_heads, head_dim, use_triton=False
    ).cuda().half()
    
    # Triton版本
    attn_triton = LlamaAttentionTriton(
        hidden_size, num_heads, num_kv_heads, head_dim, use_triton=True
    ).cuda().half()
    
    # Warmup
    for _ in range(5):
        _ = attn_torch(x)
        _ = attn_triton(x)
    torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = attn_torch(x)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000
    
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = attn_triton(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"PyTorch Attention: {pytorch_time:.3f} ms")
    print(f"Triton Attention:  {triton_time:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")


def benchmark_full_model():
    """测试完整模型性能"""
    print("\n=== Benchmark: Full Model ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    # 使用中等配置
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
    config.vocab_size = 128256
    config.hidden_size = 2048
    config.intermediate_size=8192
    config.num_hidden_layers=16
    config.num_attention_heads=16
    config.num_key_value_heads=4
    config.max_position_embeddings=8192
    
    batch_size = 1
    seq_len = 512
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
    
    # PyTorch版本
    model_torch = LlamaForCausalLM(config, use_triton=False).cuda().half()
    model_torch.eval()
    
    # Triton版本
    model_triton = LlamaForCausalLM(config, use_triton=True).cuda().half()
    model_triton.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model_torch(input_ids)
            _ = model_triton(input_ids)
    torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 5
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model_torch(input_ids)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / n_iters * 1000
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model_triton(input_ids)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"Model config: {config.num_hidden_layers} layers, {config.hidden_size} hidden_size")
    print(f"Batch size: {batch_size}, Seq length: {seq_len}")
    print(f"PyTorch Model: {pytorch_time:.3f} ms")
    print(f"Triton Model:  {triton_time:.3f} ms")
    print(f"Speedup: {pytorch_time/triton_time:.2f}x")
    
    # 计算吞吐量
    tokens_per_sec_torch = (batch_size * seq_len) / (pytorch_time / 1000)
    tokens_per_sec_triton = (batch_size * seq_len) / (triton_time / 1000)
    print(f"\nThroughput:")
    print(f"  PyTorch: {tokens_per_sec_torch:.1f} tokens/sec")
    print(f"  Triton:  {tokens_per_sec_triton:.1f} tokens/sec")


def benchmark_generation():
    """测试生成速度"""
    print("\n=== Benchmark: Text Generation ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    from interfence.engine import InferenceEngine
    
    # 使用小配置快速测试
    config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
    config.vocab_size = 128256
    config.hidden_size = 1024
    config.intermediate_size = 4096
    config.num_hidden_layers = 8
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.max_position_embeddings = 4096
    
    prompt = "Hello world, this is a test"
    max_new_tokens = 50
    
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    
    # PyTorch版本
    print("\nPyTorch version:")
    engine_torch = InferenceEngine(config=config, dtype=torch.float16)
    # 这里简化处理，实际应该运行完整生成
    
    # Triton版本
    print("\nTriton version:")
    engine_triton = InferenceEngine(config=config, dtype=torch.float16)
    # 这里简化处理，实际应该运行完整生成
    
    print("\nNote: Full generation benchmark requires trained model weights")


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("LLaMA 3 Speed Benchmark")
    print("=" * 60)
    
    benchmarks = [
        benchmark_rmsnorm,
        benchmark_ffn,
        benchmark_attention,
        benchmark_full_model,
        benchmark_generation,
    ]
    
    for benchmark in benchmarks:
        try:
            benchmark()
        except Exception as e:
            print(f"\nError in {benchmark.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_benchmarks()