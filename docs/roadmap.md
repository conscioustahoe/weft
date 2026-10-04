# Roadmap

Each milestone builds on the one before it. Each one ends with a note in [notes/](notes/) about what I learned. It also adds the web UI page that shows it off.

## Milestone 0 is a spike

No platform yet. The point is to find where the hard parts are before designing anything.

- On the Mac I load Qwen3.5-0.8B with PEFT. I write a plain PyTorch loop where forward_backward and optim_step are two separate functions. Then I train on a few hundred toy BAI2 rows.
- I save the full state and kill the process. Full state means the adapter, optimizer, step, data position and RNG. Then I resume and check that the loss picks up exactly where it left off.
- On a rented 4090 I run the same script on CUDA. I record peak memory and tokens per second. I also load 4B once to see how much memory it takes.
- I serve the base model in vLLM with LoRA turned on. I load two adapters at runtime, sample from both and get logprobs.
- I check that PEFT and vLLM handle Qwen3.5's hybrid architecture without problems. If they don't I log it in decisions.md.

What comes out is numbers and a note. After that I write docs/architecture.md.

## Milestone 1 is one job end to end

A Temporal workflow drives one LoRA training session through the primitives. It writes checkpoints to S3 style storage and registers the adapter. One vLLM server serves it. A fake worker lets the whole thing run on the Mac without a GPU.

UI page is Runs with status, step and live loss.

## Milestone 2 is multi-LoRA serving

Lots of adapters on one base model, loaded from storage when they're needed. I benchmark latency and throughput as the number of adapters grows, cold vs warm.

UI pages are Checkpoints and adapters plus the Playground.

## Milestone 3 is many tenants on one GPU that survive crashes

A job queue and a scheduler with tenants taking turns on one base model. Then preemption and resume. I kill a worker mid run and every tenant resumes from its last checkpoint with no steps lost or run twice. This runs on spot instances so the preemptions are real.

UI pages are GPU view and Recovery timeline.

## Milestone 4 is batching tenants together in training

Examples from several tenants go in one batch over the shared base weights. Each example is routed to its own adapter. I compare throughput against taking turns.

## Milestone 5 closes the loop

A job API on top of the primitives that trains, evaluates, promotes and serves on its own. Forecasting runs as the second tenant next to BAI2.

UI page is Jobs.

## Milestone 6 is RL

RL on the same primitives. The loop samples from the current adapter, scores the answers and trains on them. That needs custom losses on the server and sampling inside training. Forecasting with a Brier score reward is the first run since it has a clear reward and an SFT baseline to beat.

## Later

Tinker SDK wire compatibility, agents as a third tenant and moving the control plane to a small VM.
