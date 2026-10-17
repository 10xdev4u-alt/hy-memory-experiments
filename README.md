# Hy-Memory Experiments

Wacky experiments running Tencent's Hy-Memory-30B-A3B (30B MoE, ~3B active) on Kaggle 2x T4 GPUs.

## Kernels
- `kaggle/experiment_llamacpp.py` — llama.cpp Q4_K_M quant (rapatsky GGUF), hy_v3 support
- `kaggle/experiment_vllm.py` — vLLM + bitsandbytes 4-bit on base BF16

## Model
- https://huggingface.co/tencent/Hy-Memory-30B-A3B
- GGUF quants: https://huggingface.co/rapatsky/Hy-Memory-30B-A3B-GGUF
- Repo: https://github.com/Tencent-Hunyuan/Hy-Memory

## Status
Both kernels currently QUEUED on Kaggle (T4 allocation backlog).
