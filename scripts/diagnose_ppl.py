"""GPU diagnostic: find the per-sample PPL outliers in AISteer RepE outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2xl():
    print("[diag] loading gpt2-xl ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl", device_map="auto")
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    print("[diag] loaded.", flush=True)
    return model, tok


def ppl_batch(model, tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    with torch.no_grad():
        logits = model(ids, attention_mask=attn, labels=ids).logits.detach()
    sl = logits[..., :-1, :].contiguous()
    sla = ids[..., 1:].contiguous()
    sm = attn[..., 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss(reduction="none")(
        sl.view(-1, sl.size(-1)), sla.view(-1)
    ).view(sla.size())
    per = (loss * sm).sum(dim=1) / sm.sum(dim=1).clamp(min=1)
    return torch.exp(per).tolist()


def per_sample_ppl(model, tok, texts, bs=10):
    out = []
    for i in range(0, len(texts), bs):
        out.extend(ppl_batch(model, tok, texts[i:i + bs]))
        if (i // bs) % 10 == 0:
            print(f"  [diag] {i}/{len(texts)}", flush=True)
    return np.asarray(out)


def main():
    paths = {
        "AI-RepE": "results/raw/Llama3.1-8B-Base/l13-repe-T1.0-seed42.jsonl",
        "SRC-RepE": "/storage/work/sbd5760/odesteer/results/truthfulqa/raw_outputs/Llama3.1-8B-Base/Llama3.1-8B-Base-l13-RepE-T1.0-TruthfulQA-seed42.jsonl",
        "AI-ITI": "results/raw/Llama3.1-8B-Base/l13-iti-T1.0-seed42.jsonl",
        "SRC-ITI": "/storage/work/sbd5760/odesteer/results/truthfulqa/raw_outputs/Llama3.1-8B-Base/Llama3.1-8B-Base-l13-ITI-T1.0-TruthfulQA-seed42.jsonl",
    }
    model, tok = load_gpt2xl()
    results = {}
    for label, p in paths.items():
        if not Path(p).exists():
            print(f"[diag] missing: {p}", flush=True)
            continue
        rows = [json.loads(l) for l in open(p)]
        texts = [r["output"] for r in rows]
        questions = [r["prompt"] for r in rows]
        ppls = per_sample_ppl(model, tok, texts)
        finite = ppls[np.isfinite(ppls)]
        print(f"\n[{label}] n={len(ppls)} mean={ppls.mean():.2f} finite_mean={finite.mean():.2f} "
              f"median={np.median(finite):.2f} p95={np.percentile(finite, 95):.2f} "
              f"p99={np.percentile(finite, 99):.2f} max={finite.max():.2f}", flush=True)
        # top-5 worst
        order = np.argsort(ppls)[::-1][:5]
        for idx in order:
            print(f"  [{idx}] ppl={ppls[idx]:.1f}", flush=True)
            print(f"    Q: {questions[idx][:80]}", flush=True)
            print(f"    A: {texts[idx][:160]!r}", flush=True)
        results[label] = {
            "ppls": ppls.tolist(),
            "texts": texts,
            "questions": questions,
        }

    # If we have both AI and SRC for RepE, compare the AI top-5 worst against
    # what SRC produced for the same question.
    if "AI-RepE" in results and "SRC-RepE" in results:
        ai = results["AI-RepE"]
        src = results["SRC-RepE"]
        src_by_q = dict(zip(src["questions"], zip(src["texts"], src["ppls"])))
        ai_ppls = np.asarray(ai["ppls"])
        order = np.argsort(ai_ppls)[::-1][:5]
        print("\n[diag] AI-RepE top-5 worst, with SRC peer for same Q:", flush=True)
        for idx in order:
            q = ai["questions"][idx]
            src_text, src_ppl = src_by_q.get(q, ("?", float("nan")))
            print(f"  Q: {q[:80]}", flush=True)
            print(f"  AI  (ppl={ai_ppls[idx]:.1f}): {ai['texts'][idx]!r}", flush=True)
            print(f"  SRC (ppl={src_ppl:.1f}): {src_text!r}", flush=True)


if __name__ == "__main__":
    main()
