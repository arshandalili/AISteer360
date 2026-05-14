"""Reproduce the source repo's TruthfulQA pipeline (2-fold CV + AllenAI judges + quality)."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import seed_everything

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.evaluation.metrics.custom.truthful_qa.allenai_judges import (
    AllenAIInformativeness,
    AllenAITruthfulness,
)
from aisteer360.evaluation.metrics.custom.truthful_qa.quality import DistinctN, GPT2XLPerplexity
from aisteer360.evaluation.use_cases.truthful_qa import TruthfulQA
from aisteer360.evaluation.use_cases.truthful_qa._lm_setup import configure_for_source_repo
from aisteer360.evaluation.use_cases.truthful_qa._memory import hard_free, mem_report


METHODS = {
    "repe": "RepE", "caa": "CAA", "iti": "ITI", "mimic": "MiMiC", "lin_act": "LinAcT",
    "sphere_steer": "SphereSteer", "cobras": "COBRAS", "ode_steer": "ODESteer",
    "rff_ode_steer": "RFFODESteer", "step_ode_steer": "StepODESteer",
    "rff_step_ode_steer": "RFFStepODESteer",
}


def build_control(method: str, layer_id: int, T: float, extra: dict | None = None):
    mod = importlib.import_module(f"aisteer360.algorithms.state_control.{method}")
    kwargs = {"layer_id": layer_id, "T": T, **(extra or {})}
    return mod.STEERING_METHOD["control"](**kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-model-name", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=-1)
    ap.add_argument("--method-extra", default="", help="JSON dict of extra control kwargs.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--judge-dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    ap.add_argument("--skip-eval", action="store_true")
    return ap.parse_args()


def _load_existing(raw_path: Path, data_model_name: str, data_dir) -> list[dict]:
    from aisteer360.evaluation.use_cases.truthful_qa.data import load_tqa_answers
    with open(raw_path) as f:
        raw_rows = [json.loads(line) for line in f]
    correct_by_q, incorrect_by_q = {}, {}
    for split_idx in (0, 1):
        c, i = load_tqa_answers(split_idx, data_dir)
        correct_by_q.update(c)
        incorrect_by_q.update(i)
    out = []
    for r in raw_rows:
        q = r.get("question", r["prompt"])
        resp = r.get("response", r["output"]).split("\nQ:")[0]
        out.append({
            "question": q,
            "response": resp,
            "split": r.get("split", 0),
            "correct_answers": r.get("correct_answers") or correct_by_q.get(q, []),
            "incorrect_answers": r.get("incorrect_answers") or incorrect_by_q.get(q, []),
        })
    return out


def _generate(args, dtype: torch.dtype, raw_path: Path) -> list[dict]:
    from aisteer360.evaluation.use_cases.truthful_qa.data import load_tqa_gen_data

    extra = json.loads(args.method_extra) if args.method_extra else {}
    control = build_control(args.method, args.layer, args.T, extra)
    pipeline = SteeringPipeline(
        model_name_or_path=args.model,
        controls=[control],
        device_map="auto",
        hf_model_kwargs={"torch_dtype": dtype},
    )
    configure_for_source_repo(pipeline.model, pipeline.tokenizer)
    pos1, neg1 = load_tqa_gen_data(args.data_model_name, args.layer, 1, args.data_dir)
    pipeline.steer(pos_activations=pos1, neg_activations=neg1)

    use_case = TruthfulQA(
        model_name=args.data_model_name,
        layer_idx=args.layer,
        evaluation_metrics=[],
        data_dir=args.data_dir,
        num_samples=args.num_samples,
    )
    gens = use_case.generate(
        model_or_pipeline=pipeline,
        tokenizer=pipeline.tokenizer,
        batch_size=args.batch_size,
    )
    with open(raw_path, "w") as f:
        for g in gens:
            f.write(json.dumps({
                "prompt": g["question"],
                "output": g["response"],
                "generator": f"{args.data_model_name}-{args.method}",
                "dataset": "TruthfulQA",
                "T": args.T,
                "split": g["split"],
            }) + "\n")
    print(f"[generate] wrote {len(gens)} rows to {raw_path}")

    try:
        pipeline.model.to("meta")
    except Exception:
        pass
    del pipeline
    hard_free()
    mem_report("after-generation")
    return gens


def _evaluate(args, generations: list[dict]) -> dict:
    judge_dtype = None if args.judge_dtype == "auto" else {
        "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16
    }[args.judge_dtype]

    print(f"[eval] truthfulness ({args.judge_dtype}) ...")
    tm = AllenAITruthfulness(dtype=judge_dtype)
    truth = tm(responses=generations)
    tm.free(); del tm; hard_free(); mem_report("after-truth")

    print(f"[eval] informativeness ({args.judge_dtype}) ...")
    im = AllenAIInformativeness(dtype=judge_dtype)
    info = im(responses=generations)
    im.free(); del im; hard_free(); mem_report("after-info")

    print(f"[eval] perplexity ({args.judge_dtype}) ...")
    pm = GPT2XLPerplexity(dtype=judge_dtype)
    ppl = pm(responses=generations)
    pm.free(); del pm; hard_free(); mem_report("after-ppl")

    print("[eval] distinct-n ...")
    d1 = DistinctN(1)(responses=generations)
    d2 = DistinctN(2)(responses=generations)
    d3 = DistinctN(3)(responses=generations)
    return {"truth": truth, "info": info, "ppl": ppl, "d1": d1, "d2": d2, "d3": d3}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    raw_dir = Path(args.results_dir) / "raw" / args.data_model_name
    eval_dir = Path(args.results_dir) / "eval" / args.data_model_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"l{args.layer}-{args.method}-T{args.T}-seed{args.seed}.jsonl"
    csv_path = eval_dir / f"l{args.layer}-TruthfulQA-seed{args.seed}.csv"
    print(f"[setup] method={args.method} model={args.model} layer={args.layer} T={args.T} seed={args.seed}")
    print(f"[setup] raw -> {raw_path}")
    print(f"[setup] eval -> {csv_path}")

    if raw_path.exists():
        print(f"[generate] reusing existing {raw_path}")
        generations = _load_existing(raw_path, args.data_model_name, args.data_dir)
    else:
        generations = _generate(args, dtype, raw_path)
        generations = _load_existing(raw_path, args.data_model_name, args.data_dir)

    if args.skip_eval:
        print("[eval] skipped")
        return

    scores = _evaluate(args, generations)
    t = np.asarray(scores["truth"]["scores"])
    i = np.asarray(scores["info"]["scores"])
    row = {
        "Model": f"{args.data_model_name}-l{args.layer}",
        "Steering Method": f"{METHODS[args.method]}-T{args.T}",
        "True * Info": float((t * i).mean()) if t.size else 0.0,
        "Truthfulness": scores["truth"]["truthfulness_rate"],
        "Informativeness": scores["info"]["informativeness_rate"],
        "Perplexity": scores["ppl"]["perplexity"],
        "Dist-1": scores["d1"]["dist_1"],
        "Dist-2": scores["d2"]["dist_2"],
        "Dist-3": scores["d3"]["dist_3"],
        "n_samples": len(generations),
    }
    df_new = pd.DataFrame([row])
    df = pd.concat([pd.read_csv(csv_path), df_new], ignore_index=True) if csv_path.exists() else df_new
    df.to_csv(csv_path, index=False)
    print(f"[eval] appended row to {csv_path}")
    print(df_new.to_string(index=False))


if __name__ == "__main__":
    main()
