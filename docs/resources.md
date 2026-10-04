# Resources

Papers, docs, code and posts that are useful for building Weft. Some help directly with a piece of the system. Others are the foundations underneath it. I add to this as I go.

## Building Weft

### Training primitives

| Resource | Useful for |
| --- | --- |
| [Tinker SDK cheatsheet](https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/) | What forward_backward carries (tokens, loss targets, weights) and what it returns |
| [Tinker TrainingClient reference](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/) | How futures let a client queue forward_backward and optim_step before results come back |
| [Baseten Loops](https://docs.baseten.co/reference/sdk/loops/overview) | Another service that speaks Tinker's API |
| [Fireworks training SDK](https://docs.fireworks.ai/api-reference/training-sdk/training-and-sampling-workflow) | One more take on the same API |
| [Baseten sampler weights vs full state](https://docs.baseten.co/loops/train-on-your-data) | What a resumable checkpoint has to hold |
| [PyTorch autograd](https://pytorch.org/docs/stable/autograd.html) | How gradients build up across backward calls before an optimizer step |
| [PEFT docs](https://huggingface.co/docs/peft) | Adding LoRA adapters to a model and saving or loading just the adapter |

### LoRA

| Resource | Useful for |
| --- | --- |
| [LoRA paper](https://arxiv.org/abs/2106.09685) | The original idea and why low rank updates work |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | LoRA settings that work like rank, learning rate and which layers |
| [Tinker SL hyperparameters tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sl-hyperparams/) | Picking hyperparameters for supervised fine-tuning |

### Shared training

| Resource | Useful for |
| --- | --- |
| [mLoRA](https://arxiv.org/pdf/2312.02515) | Training several adapters in one batch over one base model with its BatchLoRA operator |
| [LobRA](https://arxiv.org/abs/2509.01193) | Scheduling tenants when sequence lengths are different |
| [tLoRA](https://arxiv.org/html/2602.07263v2) | More on scheduling many LoRA training jobs together |

### Serving

| Resource | Useful for |
| --- | --- |
| [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/) | Batching across adapters, loading adapters at runtime and the max_loras, max_cpu_loras and max_lora_rank settings |
| [Anyscale multi-LoRA guide](https://docs.anyscale.com/llm/serving/multi-lora) | Running multi-LoRA serving in practice |
| [PagedAttention (vLLM paper)](https://arxiv.org/abs/2309.06180) | How vLLM manages KV cache memory |
| [Punica](https://arxiv.org/abs/2310.18547) | Batching requests for different adapters in one kernel |
| [S-LoRA](https://arxiv.org/abs/2311.03285) | Serving thousands of adapters with adapter paging |
| [LoRAX](https://github.com/predibase/lorax) | An open source multi-LoRA server to compare against |
| [Symbiosis](https://arxiv.org/pdf/2507.03220) | Sharing a base model across training and serving plus a good related work section |

### Orchestration and durability

| Resource | Useful for |
| --- | --- |
| [Temporal docs](https://docs.temporal.io/) | Workflows, activities, retries, heartbeats and timeouts for a long lived GPU worker |
| Designing Data-Intensive Applications | Idempotency, exactly once effects and why no optimizer step should run twice |

### Measurement

| Resource | Useful for |
| --- | --- |
| vLLM benchmark scripts in the [vLLM repo](https://github.com/vllm-project/vllm) | Load generation, time to first token and inter-token latency |
| [PaLM paper](https://arxiv.org/abs/2204.02311) | Where model FLOPs utilization is defined (appendix B) |
| [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) | Memory and compute estimates for training |

### Workloads and evaluation

| Resource | Useful for |
| --- | --- |
| [Tinker forecasting recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/forecasting) | Forecasting data, prompts and evals to compare against |
| [Tinker supported models](https://thinkingmachines.ai/tinker/) | Picking a base model so the same run can be compared against real Tinker |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | Reliability diagrams and expected calibration error |
| Qwen3.5 model cards on Hugging Face | Training cutoff for the forecasting date split |

### RL

| Resource | Useful for |
| --- | --- |
| [Spinning Up in Deep RL](https://spinningup.openai.com/) | RL basics from the ground up |
| [PPO](https://arxiv.org/abs/1707.06347) | The standard policy gradient method |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO which drops the value model and fits well with sampling many answers per prompt |
| [DPO](https://arxiv.org/abs/2305.18290) | Learning from preferences without a reward model |
| [TRL docs](https://huggingface.co/docs/trl) | Reference implementations of these methods |
| [Tinker cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) | RL loops written on top of the same primitives |

## Foundations

| Resource | Useful for |
| --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | The transformer |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | A small readable GPT training loop |
| [Adam](https://arxiv.org/abs/1412.6980) and [AdamW](https://arxiv.org/abs/1711.05101) | What the optimizer state is and why it costs memory |
| [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | Whether a workload is limited by compute, memory bandwidth or overhead |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Why attention is a memory problem on GPUs |
| [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | The linear attention layers in Qwen3.5's hybrid architecture |
| [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) | How training scales across GPUs even though Weft stays on one |
