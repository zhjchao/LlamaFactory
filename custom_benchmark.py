"""Custom benchmark tool for internal hardware evaluation (Lenovo).

Simulated customization: quick latency/throughput probe used before
deploying LlamaFactory models on internal GPU nodes.
"""

import time


def benchmark_generate(model, tokenizer, prompt: str, *, max_new_tokens: int = 128,
                       warmup: int = 1, repeats: int = 3) -> dict:
    """Run a small generate benchmark, return latency/throughput stats."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for _ in range(warmup):
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        latencies.append(time.perf_counter() - t0)

    avg = sum(latencies) / len(latencies)
    return {
        "avg_latency_s": round(avg, 3),
        "min_latency_s": round(min(latencies), 3),
        "max_latency_s": round(max(latencies), 3),
        "tokens_per_s": round(max_new_tokens / avg, 1),
        "repeats": repeats,
    }


def format_report(stats: dict, model_name: str) -> str:
    """Render stats as a one-line report for internal logs."""
    return (f"[BENCH] {model_name}: avg={stats['avg_latency_s']}s "
            f"tok/s={stats['tokens_per_s']} (n={stats['repeats']})")
