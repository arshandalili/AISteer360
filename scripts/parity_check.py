"""Byte-identical check: AISteer360 SteeringPipeline+CAA vs a source-repo-style hook."""
from __future__ import annotations

import argparse
import sys

import torch
from lightning import seed_everything
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.caa import CAA
from aisteer360.algorithms.state_control.common.algorithms import CAA as CAACore
from aisteer360.evaluation.use_cases.truthful_qa._lm_setup import configure_for_source_repo


PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between weather and climate.",
    "How does a black hole form?",
    "What was the cause of World War I?",
    "Why is the sky blue?",
]
SYSTEM = (
    "Interpret each question literally, and as a question about the real world; "
    "carefully research each answer, without falling prey to any common myths; "
    "and reply 'I have no comment' unless you are completely certain of the answer."
)
GEN_CFG = GenerationConfig(
    max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9,
    repetition_penalty=1.1, use_cache=True,
)


def _resolve_layer(model, layer_id: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_id]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_id]
    raise ValueError(f"Cannot locate layer {layer_id} in {type(model).__name__}")


def _ref_hook(steer_obj, T: float):
    def hook(module, args, kwargs, output):
        hidden, rest = (output[0], output[1:]) if isinstance(output, tuple) else (output, ())
        last = hidden[:, -1, :]
        steered = steer_obj.steer(last.to(torch.float32), T=T).to(hidden.dtype)
        new = hidden.clone()
        new[:, -1, :] = steered
        return (new, *rest) if rest else new
    return hook


def _make_inputs(tokenizer, prompts, device):
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
    chat_strs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(chat_strs, return_tensors="pt", padding=True).to(device)


def _trim(tokenizer, output_ids, prompt_len=None):
    sliced = output_ids if prompt_len is None else output_ids[:, prompt_len:]
    return [r.split("\nQ:")[0] for r in tokenizer.batch_decode(sliced, skip_special_tokens=True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--num-prompts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--data-model-name", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    from aisteer360.evaluation.use_cases.truthful_qa.data import load_tqa_gen_data
    pos, neg = load_tqa_gen_data(args.data_model_name, args.layer, split_idx=0)
    prompts = PROMPTS[: args.num_prompts]

    # reference path (source-repo-style hook on a bare HF model)
    print(f"Loading reference model {args.model} ({args.dtype}) ...")
    ref_tok = AutoTokenizer.from_pretrained(args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=dtype)
    configure_for_source_repo(ref_model, ref_tok)
    steer_ref = CAACore().fit(pos.to(torch.float32), neg.to(torch.float32))
    layer = _resolve_layer(ref_model, args.layer)
    handle = layer.register_forward_hook(_ref_hook(steer_ref, args.T), with_kwargs=True)
    try:
        seed_everything(args.seed)
        inputs = _make_inputs(ref_tok, prompts, ref_model.device)
        ref_ids = ref_model.generate(**inputs, generation_config=GEN_CFG)
        ref_outs = _trim(ref_tok, ref_ids, prompt_len=inputs.attention_mask.shape[1])
    finally:
        handle.remove()
    del ref_model
    torch.cuda.empty_cache()

    # AISteer360 path (hook applied via SteeringPipeline's state_control context)
    print("Loading AISteer360 SteeringPipeline ...")
    pipeline = SteeringPipeline(
        model_name_or_path=args.model,
        controls=[CAA(layer_id=args.layer, T=args.T)],
        device_map="auto",
        hf_model_kwargs={"torch_dtype": dtype},
    )
    configure_for_source_repo(pipeline.model, pipeline.tokenizer)
    pipeline.steer(pos_activations=pos, neg_activations=neg)
    seed_everything(args.seed)
    inputs = _make_inputs(pipeline.tokenizer, prompts, pipeline.model.device)
    ai_ids = pipeline.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=GEN_CFG,
    )
    ai_outs = _trim(pipeline.tokenizer, ai_ids)

    fail = False
    for i, (p, r, a) in enumerate(zip(prompts, ref_outs, ai_outs)):
        ok = r == a
        print(f"\n[{i}] {'OK' if ok else 'DIFF'}: {p}")
        print(f"  ref: {r!r}")
        print(f"  ai : {a!r}")
        if not ok:
            fail = True

    print("\nParity FAILED" if fail else "\nParity OK")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
