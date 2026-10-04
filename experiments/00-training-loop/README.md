# 00 Training loop

A plain PyTorch and PEFT training loop for Qwen3.5-0.8B with no platform around it. forward_backward and optim_step are two separate functions. It saves the full training state, gets killed and resumes exactly where it left off.

## Steps

| Script | What it does |
| --- | --- |
| `load_and_generate.py` | Loads the model, shows how text turns into tokens and generates once |

## Run

```bash
uv run python experiments/00-training-loop/load_and_generate.py
```

## Results

| What | Mac M4 16 GB |
| --- | --- |
| Parameters | 752M |
| Load time from cache | 2.4s |
| Generation speed with the reference kernels | about 11 tokens/s |

## Findings

- transformers loads the text only part of the model as `Qwen3_5ForCausalLM`. The vision part is skipped.
- Qwen3.5's linear attention layers fall back to plain PyTorch code on the Mac because the fast kernels (flash-linear-attention and causal-conv1d) need CUDA. It works but it's slow.
- The tokenizer splits numbers into single digits. An amount like 12500 is five tokens.
- Asked what a BAI2 file is the base model makes up an answer. It calls it a binary format which it isn't.
