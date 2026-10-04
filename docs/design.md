# Design

## Summary

Weft is a small service for fine-tuning and serving open models with LoRA. Users write their own training loop and call the service for forward and backward passes, optimizer steps, sampling and checkpoints. The service owns the GPU and keeps one copy of the base model that all tenants share. It serves every tenant's adapters from one inference server.

There's also a simple job API on top. It's for people who just want to upload data and get a model back.

## The demo

The finished project is one recorded demo. Two tenants train and serve on one rented GPU. A crash happens and both recover. Then their adapters get served side by side.

1. Tenant A installs a small Python client on a laptop and points it at the service.
2. Tenant A creates a training client for a base model with a LoRA rank. The service sets up that tenant's adapter and optimizer state on the GPU.
3. Tenant A runs an SFT loop over synthetic BAI2 rows and calls forward_backward and optim_step for each batch. The loop runs on the laptop. All the GPU work happens on the rented box.
4. Tenant B starts a forecasting run on the same base model at the same time. Both share one copy of the base weights and the service decides whose step runs next.
5. Halfway through I kill the GPU worker. Orchestration notices and restarts it. Both tenants come back from their last checkpoints. Both loops keep going with no steps lost or run twice. The web UI shows the recovery as it happens.
6. Each tenant saves weights for sampling. The adapter gets loaded into the serving layer without a restart.
7. Both tenants call sample with their own adapter and get answers from the same inference server.
8. A third user doesn't write a loop at all. They submit a job from the web UI with a dataset and a base model. The job runs the same loop on the server and returns an adapter ID.

## The primitives API

A few calls make up the whole training and sampling API. Everything else is built from them and that includes jobs. The names match [Tinker's SDK](https://tinker-docs.thinkingmachines.ai/tinker/sdk-cheatsheet/). [Baseten](https://docs.baseten.co/reference/sdk/loops/overview) and [Fireworks](https://docs.fireworks.ai/api-reference/training-sdk/training-and-sampling-workflow) use the same names too.

| Call | What the user sends | What comes back | Why it exists |
| --- | --- | --- | --- |
| create_lora_training_client | Base model and LoRA rank | Training client | Sets up a tenant's adapter and optimizer state |
| forward_backward | Batch of examples and loss type | Loss and metrics | Computes gradients on the server and adds them up so several calls can make one step |
| optim_step | Adam settings | Step number | Applies the gradients added up so far |
| save_state / load_state_with_optimizer | Checkpoint name or path | Checkpoint path | Saves everything needed to resume including the optimizer |
| save_weights_for_sampler | Name | Sampler weights path | Exports just the adapter and registers it for serving |
| sample | Adapter path, prompt and sampling settings | Completions | Runs inference with any saved adapter |
| compute_logprobs | Adapter path and token sequence | Log probability per token | Scores text like the chance of Yes for forecasting |

Calls return futures. That way a client can send the next batch while the current step is still running. The loss function lives on the server for now. Custom losses come later with RL.

## The job layer

A job is a durable workflow that runs a normal training loop on the server. It only uses the public primitives. If I can write a job with nothing but those calls then the primitives have everything they need.

A job takes a dataset, a base model, hyperparameters and an eval set. It checks the data and calls create_lora. Then it loops over forward_backward and optim_step and calls save_state every N steps. At the end it runs the eval, calls save_weights_for_sampler and returns the adapter ID. Since jobs are built on the primitives any fix or speedup to the primitives makes jobs better too.

## Web UI

The web UI shows multi-tenancy, crashes and recovery so you don't have to dig through logs. It uses the same APIs as the Python client with no special access.

- **Runs** shows every tenant's training runs with status, current step and live loss curves
- **GPU view** shows memory used by the base weights and by each tenant, utilization and whose step is running now
- **Checkpoints and adapters** shows saved states, sampler weights and which adapters are loaded for serving
- **Playground** lets you pick an adapter and send a prompt. It shows the output and token probabilities next to the base model's
- **Jobs** lets you submit a job by uploading a dataset and picking a base model and settings
- **Recovery timeline** shows a worker dying and the run being detected, restored and resumed live

## First tenant is BAI2 extraction

BAI2 is a file format banks use for transaction reports. The first workload pulls structured fields out of BAI2 transaction records. It trains on synthetic data and gets tested on real public samples. Jobs are short and cheap and scoring is exact match. So it tests the system without needing a complicated dataset.

- **Input** is one transaction plus some context like type code, amount, date and bank. A transaction is a type 16 record and any type 88 continuation records after it.
- **Output** is JSON with sender info, receiver info, payment type and direction. Payment type is ACH, wire, RTP, push to card, check or other. Direction is credit or debit.
- **Training data** comes from a generator. It has one profile per bank because every bank writes BAI2 a bit differently. The profiles are based on public BAI2 guides from banks and a list of the ways files vary. The generator also makes the labels. No private or customer data is used.
- **Evaluation** is exact match per field for the base model and the fine-tuned model. There are three eval sets. Held out synthetic rows, 50 to 100 public sample records I label by hand and one bank profile that's left out of training completely.

## Second tenant is forecasting

The second workload is yes or no forecasting. It starts with plain SFT. It uses the same primitives as BAI2 but the data looks very different. That's what makes the multi-tenancy test meaningful.

- **Task** is to answer Yes or No given a question, how it gets resolved and background info up to a cutoff date. The forecast is the probability the model gives the Yes token.
- **Why SFT is enough to start.** Training with cross entropy on the real answer is the same as minimizing log loss. Log loss rewards honest probabilities so the model learns to be calibrated. RL with a Brier score reward comes after that.
- **Data** is a public openly licensed set of resolved questions. Something like the Prophet Arena subset from Tinker's forecasting recipe or ForecastBench. It gets split by date and evaluated only on questions that resolved after the base model's training cutoff.
- **Evaluation** is Brier score and log loss for the base and fine-tuned models plus a calibration curve. The baselines are always saying 50% and always saying the training set's base rate.

For the system it's different from BAI2 in a few ways. Prompts are long and the answer is one token. BAI2 has short prompts and a JSON answer. There are fewer examples and each one is longer. The eval needs token probabilities so sample has to be able to return logprobs.

## Metrics and charts

I log every metric below from day one. The writeup leads with the five charts marked headline. Everything runs on the same rented GPU type so numbers compare across milestones.

### Model quality

| Metric | What it means | Chart |
| --- | --- | --- |
| Field exact match (BAI2) | Share of rows where each output field matches its label | **Headline.** Grouped bars with fields on x and base vs fine-tuned. One panel per eval set |
| JSON validity (BAI2) | Share of outputs that parse and match the schema | One number next to the headline chart |
| Made up values (BAI2) | Share of predicted sender or receiver values that aren't in the input | Bars for base vs fine-tuned |
| Brier score and log loss (forecasting) | Mean squared error and mean negative log likelihood of the Yes probability | **Headline.** Bars for base, fine-tuned, the 50% baseline and the base rate baseline |
| Calibration (forecasting) | Predicted probability buckets vs how often Yes actually happened plus expected calibration error | Reliability diagram |
| Train and eval loss | Loss per step for each tenant | One line per tenant over steps |

### Shared training

| Metric | What it means | Chart |
| --- | --- | --- |
| Total throughput | Training tokens per second across all tenants | **Headline.** Line with tenants (1, 2, 4, 8) on x. Shared base vs one process per tenant |
| Step latency | Time the client sees for forward_backward plus optim_step at p50 and p95 | Line vs number of tenants |
| GPU memory per tenant | Extra memory each new tenant adds on top of the base weights | Stacked bars of base weights vs per tenant state |
| GPU utilization | How much of the time the GPU is busy and the model FLOPs utilization | Line over a run |
| Fairness | Each tenant's share of GPU time compared to an equal share | One line per tenant over time |
| Queue wait | How long a primitive call waits before it runs | p50 and p95 vs number of tenants |

### Multi-LoRA serving

| Metric | What it means | Chart |
| --- | --- | --- |
| Time to first token | Time from sending a request to getting the first token back at p50 and p95 | **Headline.** Line with active adapters (1 to 64) on x. Cold vs warm |
| Inter-token latency | Time between output tokens at p50 and p95 | Same axes as above |
| Output throughput | Output tokens per second across all requests | Line vs active adapters |
| Adapter load time | Time to load an adapter from storage when cold or from cache when warm | Bars by LoRA rank |
| Adapter cache hit rate | Share of requests that hit an adapter that's already loaded | Line vs active adapters |

### Durability and cost

| Metric | What it means | Chart |
| --- | --- | --- |
| Recovery time | Time from the worker getting killed to the failure being noticed, state restored and the first step running again | **Headline.** Timeline of one kill with a lane per tenant |
| Steps lost or run twice | Optimizer steps missing or applied twice after recovery. Should be zero | One number per kill test |
| Checkpoint cost | Time and size to save the full training state | Bars by LoRA rank |
| Cost per run | GPU price per hour times wall clock time. Per training run and per 1M training tokens | Table in the writeup |
| Cost per 1,000 inferences | GPU price per hour divided by requests served per hour at the target latency | Table in the writeup |

## Not doing

If the demo doesn't need it then it's out.

- Multi-GPU training of any kind
- Full fine-tuning since it's LoRA only
- Base models bigger than about 8B
- Billing, quotas and any auth beyond one API key per tenant
- Production level security and isolation
