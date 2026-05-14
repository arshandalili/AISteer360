"""Print everything that could affect generation, for both AI and SRC paths."""
from __future__ import annotations

import os
import sys
import json

import torch
from lightning import seed_everything


def dump_state(tag: str, model, tokenizer):
    print(f"\n[{tag}] device_map = {model.hf_device_map if hasattr(model, 'hf_device_map') else 'n/a'}", flush=True)
    print(f"[{tag}] model.device = {model.device}", flush=True)
    print(f"[{tag}] model.dtype = {model.dtype}", flush=True)
    print(f"[{tag}] model.config.eos_token_id = {model.config.eos_token_id}", flush=True)
    print(f"[{tag}] model.config.pad_token_id = {model.config.pad_token_id}", flush=True)
    print(f"[{tag}] model.config.use_cache = {model.config.use_cache}", flush=True)
    print(f"[{tag}] model.config._attn_implementation = {getattr(model.config, '_attn_implementation', 'n/a')}", flush=True)
    print(f"[{tag}] model.generation_config.eos_token_id = {model.generation_config.eos_token_id}", flush=True)
    print(f"[{tag}] tokenizer.padding_side = {tokenizer.padding_side}", flush=True)
    print(f"[{tag}] tokenizer.pad_token_id = {tokenizer.pad_token_id}", flush=True)
    print(f"[{tag}] tokenizer.chat_template hash = {hash(tokenizer.chat_template or '')}", flush=True)
    # check first layer device
    if hasattr(model.model, "layers"):
        print(f"[{tag}] layers[0].device = {next(model.model.layers[0].parameters()).device}", flush=True)
        print(f"[{tag}] layers[13].device = {next(model.model.layers[13].parameters()).device}", flush=True)
        print(f"[{tag}] layers[-1].device = {next(model.model.layers[-1].parameters()).device}", flush=True)


def get_first_logit_diff(tag: str, model, tokenizer):
    """Run a single forward pass with a fixed prompt and dump the top-5 logits at position -1."""
    seed_everything(42)
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    last_logits = out.logits[0, -1].float()
    top5 = torch.topk(last_logits, k=5)
    print(f"\n[{tag}] FIRST-FORWARD top-5 logits at position -1:", flush=True)
    for tid, lg in zip(top5.indices.tolist(), top5.values.tolist()):
        tok = tokenizer.decode([tid])
        print(f"  token_id={tid:>7d}  logit={lg:+.6f}  tok={tok!r}", flush=True)


def main():
    sys.path.insert(0, "/storage/work/sbd5760/odesteer/src")
    MODEL = "meta-llama/Llama-3.1-8B"

    # ---- AISteer SteeringPipeline ----
    from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
    from aisteer360.algorithms.state_control.caa import CAA
    from aisteer360.evaluation.use_cases.truthful_qa._lm_setup import configure_for_source_repo

    seed_everything(42)
    pipeline = SteeringPipeline(
        model_name_or_path=MODEL,
        controls=[CAA(layer_id=13, T=5.0)],
        device_map="auto",
        hf_model_kwargs={"torch_dtype": torch.float32},
    )
    configure_for_source_repo(pipeline.model, pipeline.tokenizer)
    dump_state("AI", pipeline.model, pipeline.tokenizer)
    get_first_logit_diff("AI", pipeline.model, pipeline.tokenizer)

    del pipeline
    torch.cuda.empty_cache()

    # ---- Source HuggingFaceLM ----
    from odesteer.lm import HuggingFaceLM
    seed_everything(42)
    src_lm = HuggingFaceLM(
        MODEL, "CAA",
        steer_model_kwargs={},
        steer_layer_idx=13,
        device="auto", dtype=torch.float32,
    )
    dump_state("SRC", src_lm.model, src_lm.tokenizer)
    get_first_logit_diff("SRC", src_lm.model, src_lm.tokenizer)


if __name__ == "__main__":
    main()
