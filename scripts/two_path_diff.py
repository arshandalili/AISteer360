"""Run source-repo-style flow and AISteer flow in the same process, same RNG, same prompts."""
from __future__ import annotations

import argparse
from functools import partial

import torch
from lightning import seed_everything
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.caa import CAA
from aisteer360.algorithms.state_control.common.algorithms import CAA as CAACore
from aisteer360.evaluation.use_cases.truthful_qa._lm_setup import (
    DEFAULT_CHAT_TEMPLATE,
    configure_for_source_repo,
)


SYSTEM = (
    "Interpret each question literally, and as a question about the real world; "
    "carefully research each answer, without falling prey to any common myths; "
    "and reply 'I have no comment' unless you are completely certain of the answer."
)


def src_register(model, layer_idx, steer_obj, T):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, ()
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        hidden = hidden.clone()
        hidden[batch_idx, -1] = steer_obj.steer(hidden[batch_idx, -1], T=T)
        return (hidden, *rest) if rest else hidden
    layer = model.model.layers[layer_idx]
    return layer.register_forward_hook(hook)


def src_generate(model, tok, prompts, gen_cfg):
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
    chat_strs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat_strs, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, generation_config=gen_cfg)
    prompt_len = inputs.attention_mask.shape[1]
    raw = tok.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
    return [r.split("\nQ:")[0] for r in raw]


def ai_generate(pipeline, prompts, gen_cfg):
    tok = pipeline.tokenizer
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
    chat_strs = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat_strs, return_tensors="pt", padding=True).to(pipeline.model.device)
    out_ids = pipeline.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        generation_config=gen_cfg,
    )
    raw = tok.batch_decode(out_ids, skip_special_tokens=True)
    return [r.split("\nQ:")[0] for r in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--data-model-name", required=True)
    ap.add_argument("--T", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-prompts", type=int, default=8)
    args = ap.parse_args()

    from aisteer360.evaluation.use_cases.truthful_qa.data import (
        load_tqa_gen_data,
        load_tqa_gen_questions,
    )
    pos, neg = load_tqa_gen_data(args.data_model_name, args.layer, 1)
    prompts = load_tqa_gen_questions(0)[: args.num_prompts]

    gen_cfg = GenerationConfig(
        max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.1, use_cache=True,
    )

    print("\n=== SRC-STYLE PATH ===", flush=True)
    seed_everything(args.seed)
    tok_a = AutoTokenizer.from_pretrained(args.model)
    model_a = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float32)
    configure_for_source_repo(model_a, tok_a)
    steer_a = CAACore().fit(pos.to(torch.float32), neg.to(torch.float32))
    h = src_register(model_a, args.layer, steer_a, args.T)
    seed_everything(args.seed)
    src_outs = src_generate(model_a, tok_a, prompts, gen_cfg)
    h.remove()
    del model_a, tok_a
    torch.cuda.empty_cache()

    print("\n=== AI-PIPELINE PATH ===", flush=True)
    seed_everything(args.seed)
    pipeline = SteeringPipeline(
        model_name_or_path=args.model,
        controls=[CAA(layer_id=args.layer, T=args.T)],
        device_map="auto",
        hf_model_kwargs={"torch_dtype": torch.float32},
    )
    configure_for_source_repo(pipeline.model, pipeline.tokenizer)
    pipeline.steer(pos_activations=pos, neg_activations=neg)
    seed_everything(args.seed)
    ai_outs = ai_generate(pipeline, prompts, gen_cfg)

    print("\n=== DIFF ===", flush=True)
    for i, (p, s, a) in enumerate(zip(prompts, src_outs, ai_outs)):
        ok = s == a
        print(f"\n[{i}] {'OK' if ok else 'DIFF'}: {p[:80]}", flush=True)
        if not ok:
            print(f"  SRC: {s!r}", flush=True)
            print(f"  AI : {a!r}", flush=True)


if __name__ == "__main__":
    main()
