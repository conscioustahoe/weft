"""Step 1. Load the base model, look at how text becomes tokens and generate once.

Run it with
    uv run python experiments/00-training-loop/load_and_generate.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    device = pick_device()
    print(f"device: {device}")

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"loaded {MODEL} in {time.perf_counter() - start:.1f}s")
    print(f"model class: {type(model).__name__}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # Text in, token ids out. Each id is one row in the model's vocabulary.
    text = "16,195,12500,0,,,ACH CREDIT ACME PAYROLL/"
    ids = tokenizer(text)["input_ids"]
    print(f"\ntext: {text!r}")
    print(f"{len(ids)} tokens: {ids}")
    print(f"pieces: {tokenizer.convert_ids_to_tokens(ids)}")

    # The chat template wraps messages in the special tokens the model was trained on.
    messages = [{"role": "user", "content": "In one sentence, what is a BAI2 file?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    print(f"\nprompt after chat template:\n{prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    elapsed = time.perf_counter() - start

    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    print(f"answer: {tokenizer.decode(new_tokens, skip_special_tokens=True)}")
    print(f"\n{len(new_tokens)} new tokens in {elapsed:.1f}s ({len(new_tokens) / elapsed:.1f} tokens/s)")


if __name__ == "__main__":
    main()
