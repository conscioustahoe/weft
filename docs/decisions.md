# Decisions

Rows are in the order I made them. When a decision changes I add a new row that replaces the old one instead of editing it. That way the history stays readable.

| Date | Area | Decision | Why |
| --- | --- | --- | --- |
| 2026-10-03 | Naming | No version labels in the plan. The project is just called Mini Tinker. Replaced by the next row. | Version labels would get confused with code releases later |
| 2026-10-03 | Naming | The project is called Weft. Replaces the Mini Tinker row above. | I wanted a name of its own instead of one borrowed from Tinker |
| 2026-10-03 | Base model | Build with Qwen3.5-0.8B and run final results and the Tinker comparison on Qwen3.5-4B. The model is a config value. | Small models are fast to iterate on and leave room on a 24 GB card. The cost difference is cents per run. 4B is on Tinker's supported list. I'll revisit if the spike shows LoRA tooling struggles with Qwen3.5's hybrid architecture |
| 2026-10-03 | Protocol | My own API using Tinker's ideas and names. Full Tinker SDK compatibility comes later. | Leaves the door open to running cookbook recipes as is without boxing in the design now |
| 2026-10-03 | Shared training | Tenants take turns on one base model first. Putting several tenants in one batch is its own milestone later. | Taking turns is simple and already saves the memory. Batching needs each example routed to its own adapter. Comparing the two makes a good chart |
| 2026-10-03 | GPU layout | Training and serving share one GPU during development. They get separate GPUs for benchmark runs. | Cheap while building and clean numbers when it counts |
| 2026-10-03 | Transport | HTTP with FastAPI. Clients send batches that are already tokenized. | Easy to debug and the web UI can call it directly. It matches Tinker. Sending tokens gives users control over chat templates and loss masks |
| 2026-10-03 | Storage | S3 style storage from day one behind a small interface. MinIO locally. | Checkpoints have to survive the GPU box dying or the recovery demo doesn't work |
| 2026-10-03 | Control plane | Runs on my Mac during development. The GPU box only runs the worker and vLLM and they talk over Tailscale. It moves to a small VM later. | The thing that notices a dead worker can't die with it |
| 2026-10-03 | Web UI | In scope but kept simple. It uses the same APIs as the Python client and grows a page with each milestone. | Makes multi-tenancy and recovery easy to see without turning into its own project |
| 2026-10-03 | Architecture | Build Tinker style training primitives first. The job API sits on top of them. | Jobs show the primitives are complete and RL needs the same primitives |
| 2026-10-03 | Workloads | BAI2 extraction on synthetic data is the first tenant. Forecasting is the second. Agents come later. | BAI2 is simple and scored by exact match. Forecasting looks very different so running both is a real multi-tenancy test |
| 2026-10-03 | Training method | SFT first and RL after that. Both are part of the project. | SFT covers both tenants and keeps the first milestones simple. RL adds sampling inside training and custom losses which is a big step on its own. It runs on the same primitives |
| 2026-10-03 | Data | Only public and synthetic data. No private or customer data. | Anyone can reproduce the results and there are no IP concerns |
| 2026-10-03 | Compute | Rented cloud GPUs for real runs and the Mac for development. | Short bursts are cheap and the benchmark numbers come from the NVIDIA stack that matters |
