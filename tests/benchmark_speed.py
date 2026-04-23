"""
速度基准测试
对比PyTorch和Triton实现的性能
"""

import torch
import time
import sys
import os

# 添加父目录到路径
from safetensors.torch import load_file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llama3.model.config import LlamaConfig
from modelscope import snapshot_download
from transformers import AutoTokenizer
from llama3.model.modling_llama import LlamaForCausalLM


def benchmark_full_model():
    """测试完整模型性能"""
    print("\n=== Benchmark: Full Model ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        return
    
    # 使用中等配置
    model_dir = snapshot_download("LLM-Research/Llama-3.2-1B")
    config = LlamaConfig.from_pretrained(model_dir)
    
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
    
    from llama3.interfence.engine import InferenceEngine
    
    # 加载模型配置
    model_dir = snapshot_download("LLM-Research/Llama-3.2-1B")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = LlamaConfig.from_pretrained(model_dir)
    model_triton = LlamaForCausalLM(config, use_triton=True)
    model_torch = LlamaForCausalLM(config, use_triton=False)
    weight_path = f"{model_dir}/model.safetensors"
    state_dict = load_file(weight_path)  # 读取safetensors权重字典
    
    prompt = "Hello"
    max_new_tokens = 5
    
    print(f"Prompt: '{prompt}'")
    print(f"Max new tokens: {max_new_tokens}")
    
    def run_engine(engine, name):
        print(f"\n{name}:")
        
        # Warmup
        with torch.no_grad():
            _ = engine.generate(prompt, max_new_tokens=5, do_sample=False)
        torch.cuda.synchronize()
        
        # 正式测试
        import time
        start = time.perf_counter()
        
        with torch.no_grad():
            output = engine.generate(
                prompt, 
                max_new_tokens=max_new_tokens,
                do_sample=False,        # 固定 greedy，保证两次输出一致且可比
                temperature=1.0,
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # 统计
        stats = engine.get_stats()
        prompt_tokens = len(tokenizer.encode(prompt))
        total_tokens = len(tokenizer.encode(output))
        new_tokens = total_tokens - prompt_tokens
        
        print(f"  Output: {output[:100]}{'...' if len(output)>100 else ''}")
        print(f"  Total time: {elapsed*1000:.2f} ms")
        print(f"  New tokens: {new_tokens}")
        print(f"  Prefill: {stats['prefill_time_ms']:.2f} ms")
        print(f"  Decode:  {stats['decode_time_ms']:.2f} ms")
        if stats.get('decode_tokens_per_sec'):
            print(f"  Decode speed: {stats['decode_tokens_per_sec']:.2f} tokens/sec")
        
        return elapsed, output
    
    # PyTorch版本
    model_torch.load_state_dict(state_dict, strict=True)
    model_torch.eval()
    engine_torch = InferenceEngine(model=model_torch, tokenizer=tokenizer, device="cuda", dtype=torch.float16)
    time_torch, out_torch = run_engine(engine_torch, "PyTorch")
    
    # 清显存，防止 Triton 测试受干扰
    del engine_torch, model_torch
    torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    
    # # Triton版本
    # model_triton.load_state_dict(state_dict, strict=True)
    # model_triton.eval()
    # engine_triton = InferenceEngine(model=model_triton, tokenizer=tokenizer, device="cuda", dtype=torch.float16)
    # time_triton, out_triton = run_engine(engine_triton, "Triton")
    
    # # 一致性检查
    # print(f"\n{'='*40}")
    # print(f"Speedup (total): {time_torch/time_triton:.2f}x")
    # if out_torch == out_triton:
    #     print("✓ Outputs match exactly")
    # else:
    #     print("⚠ Outputs differ (expected for fp16 + different kernels)")


def run_all_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("LLaMA 3 Speed Benchmark")
    print("=" * 60)
    
    benchmarks = [
        # benchmark_full_model,
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