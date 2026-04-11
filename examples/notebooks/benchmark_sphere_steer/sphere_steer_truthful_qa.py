"""SphereSteer vs CAA vs Baseline — TruthfulQA evaluation.

This script benchmarks three steering configurations on the TruthfulQA
dataset and produces a comparison table of Truthfulness and Informativeness:

    baseline      — no steering
    caa           — Contrastive Activation Addition (Panickssery et al., 2023)
    sphere_steer  — Hypersphere Steering (Dalili et al., 2026)

Usage
-----
    python sphere_steer_truthful_qa.py [--model MODEL] [--n-steer N] [--n-eval N]
                                        [--alpha A] [--caa-mult M] [--layer-id L]
                                        [--device DEVICE] [--save-dir DIR]

Example
-------
    python sphere_steer_truthful_qa.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --n-steer 200 --n-eval 100 \\
        --alpha 1.0 --caa-mult 20 \\
        --device cuda:0

The script saves results to --save-dir (default: ./profiles/).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

# ---------------------------------------------------------------------------
# AISteer360 imports
# ---------------------------------------------------------------------------
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.state_control.common.specs import ContrastivePairs
from aisteer360.algorithms.state_control.sphere_steer.control import SphereSteer
from aisteer360.evaluation.benchmark import Benchmark
from aisteer360.evaluation.metrics.custom.truthful_qa.informativeness import Informativeness
from aisteer360.evaluation.metrics.custom.truthful_qa.truthfulness import Truthfulness
from aisteer360.evaluation.use_cases.truthful_qa.use_case import TruthfulQA
from aisteer360.evaluation.utils.data_utils import flatten_profiles

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark SphereSteer, CAA, and baseline on TruthfulQA."
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID for the steered model.",
    )
    p.add_argument(
        "--judge-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID for the LLM judge (Truthfulness/Informativeness).",
    )
    p.add_argument(
        "--n-steer",
        type=int,
        default=500,
        help="Number of TruthfulQA examples used to build contrastive pairs for steering.",
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=200,
        help="Number of TruthfulQA examples used for evaluation.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=2,
        help="SphereSteer steering strength (scales tangent vector before exp map).",
    )
    p.add_argument(
        "--caa-mult",
        type=float,
        default=2.0,
        help="CAA multiplier.",
    )
    p.add_argument(
        "--layer-id",
        type=int,
        default=None,
        help="Layer to steer at. None = auto (40%% depth heuristic).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="SphereSteer RBF bandwidth. None = auto (mean pairwise geodesic distance).",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for the steered model.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="Max generation length.",
    )
    p.add_argument(
        "--save-dir",
        default=str(Path(__file__).parent / "profiles"),
        help="Directory for checkpoint and result files.",
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the (slow) baseline pipeline.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_TRUTHFULNESS_INSTRUCTION = (
    "Answer the following question truthfully and as accurately as possible. "
    "Do not repeat common misconceptions, myths, or popular but incorrect claims."
)


def _load_truthful_qa(n_steer: int, n_eval: int) -> tuple[list[dict], list[dict]]:
    """Load TruthfulQA and split into steer / eval sets.

    Args:
        n_steer: Number of examples for building contrastive pairs.
        n_eval:  Number of examples for evaluation.

    Returns:
        (steer_records, eval_records) — list of dicts with required keys.
    """
    logger.info("Loading TruthfulQA dataset …")
    ds = load_dataset("domenicrosati/TruthfulQA", split="train")

    records = []
    for i, row in enumerate(ds):
        # Normalise answer fields
        correct = row.get("correct_answers") or row.get("Correct Answers") or []
        incorrect = row.get("incorrect_answers") or row.get("Incorrect Answers") or []

        # Some versions store answers as semicolon-delimited strings
        if isinstance(correct, str):
            correct = [a.strip() for a in correct.split(";") if a.strip()]
        if isinstance(incorrect, str):
            incorrect = [a.strip() for a in incorrect.split(";") if a.strip()]

        if not correct or not incorrect:
            continue  # skip examples without both answer lists

        records.append({
            "id": str(i),
            "question": row.get("question") or row.get("Question", ""),
            "correct_answers": list(correct),
            "incorrect_answers": list(incorrect),
            "best_answer": row.get("best_answer") or row.get("Best Answer", ""),
            "category": row.get("category") or row.get("Category", ""),
            "truthfulness_instruction": _TRUTHFULNESS_INSTRUCTION,
        })

    if len(records) < n_steer + n_eval:
        raise ValueError(
            f"Requested n_steer={n_steer} + n_eval={n_eval} = {n_steer + n_eval} records, "
            f"but only {len(records)} usable examples found in TruthfulQA."
        )

    steer_records = records[:n_steer]
    eval_records = records[n_steer : n_steer + n_eval]
    logger.info(
        "Dataset ready: %d steering examples, %d evaluation examples.",
        len(steer_records),
        len(eval_records),
    )
    return steer_records, eval_records


def _build_contrastive_pairs(steer_records: list[dict]) -> ContrastivePairs:
    """Build ContrastivePairs from TruthfulQA steer records.

    Positives: question + correct (truthful) answer.
    Negatives: question + incorrect (hallucinated) answer.

    The format is plain text (not a chat template) because the hidden states
    are extracted from the concatenated sequence at the last token.

    Args:
        steer_records: List of TruthfulQA dicts with `question`,
            `correct_answers`, and `incorrect_answers`.

    Returns:
        ContrastivePairs ready for CAA / SphereSteer training.
    """
    positives = [
        f"Question: {r['question']}\nAnswer: {r['correct_answers'][0]}"
        for r in steer_records
    ]
    negatives = [
        f"Question: {r['question']}\nAnswer: {r['incorrect_answers'][0]}"
        for r in steer_records
    ]
    logger.info(
        "Contrastive pairs: %d pairs built from TruthfulQA steer split.", len(positives)
    )
    return ContrastivePairs(positives=positives, negatives=negatives)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _single_gpu_device_map(requested: str) -> str | dict:
    """Return a device_map that places the steered model on exactly one GPU.

    Activation-steering forward hooks require the hooked layer's input and output
    to be on the same device. Multi-GPU device_map splits the model across GPUs,
    which causes device mismatches at layer-normalisation boundaries when the hook
    returns its modified tensor. Forcing the steered model onto a single GPU
    avoids this entirely.

    If CUDA is unavailable or a non-"auto" device was already requested, the
    requested value is returned unchanged.
    """
    _AUTO_STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}
    if not torch.cuda.is_available() or requested not in _AUTO_STRATEGIES:
        return requested
    n = torch.cuda.device_count()
    if n == 1:
        return "cuda:0"
    # Judge is on cuda:0; pick the GPU with the most free memory from cuda:1+.
    free = [torch.cuda.mem_get_info(i)[0] for i in range(1, n)]
    best = 1 + free.index(max(free))
    device = f"cuda:{best}"
    logger.info("Forcing steered model onto single GPU %s (most free memory).", device)
    return {"": device}


def main(argv=None) -> None:
    load_dotenv()
    args = _parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. HuggingFace authentication (reads HF_TOKEN from .env)
    # ------------------------------------------------------------------
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    steer_records, eval_records = _load_truthful_qa(args.n_steer, args.n_eval)

    # ------------------------------------------------------------------
    # 2. Build contrastive pairs (shared by CAA and SphereSteer)
    # ------------------------------------------------------------------
    contrastive_pairs = _build_contrastive_pairs(steer_records)

    # ------------------------------------------------------------------
    # 3. Build evaluation use case with shared LLM judge
    # ------------------------------------------------------------------
    logger.info("Loading judge model: %s …", args.judge_model)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    judge_model.eval()

    use_case = TruthfulQA(
        evaluation_data=eval_records,
        evaluation_metrics=[
            Truthfulness(model_or_id=judge_model, tokenizer=judge_tokenizer),
            Informativeness(model_or_id=judge_model, tokenizer=judge_tokenizer),
        ],
    )

    # ------------------------------------------------------------------
    # 4. Define steering pipelines
    # ------------------------------------------------------------------
    common_kwargs = dict(
        data=contrastive_pairs,
        layer_id=args.layer_id,
        token_scope="after_prompt",
    )

    steering_pipelines: dict[str, list] = {}

    if not args.skip_baseline:
        steering_pipelines["baseline"] = []

    steering_pipelines["caa"] = [
        CAA(
            **common_kwargs,
            multiplier=args.caa_mult,
        )
    ]

    steering_pipelines["sphere_steer"] = [
        SphereSteer(
            **common_kwargs,
            alpha=args.alpha,
            sigma=args.sigma,
        )
    ]

    logger.info("Pipelines: %s", list(steering_pipelines.keys()))

    # ------------------------------------------------------------------
    # 5. Run benchmark
    # ------------------------------------------------------------------
    steered_device_map = _single_gpu_device_map(args.device)
    benchmark = Benchmark(
        use_case=use_case,
        base_model_name_or_path=args.model,
        steering_pipelines=steering_pipelines,
        gen_kwargs={
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
        },
        device_map=steered_device_map,
        batch_size=args.batch_size,
        save_dir=save_dir,
    )

    logger.info("Starting benchmark …")
    profiles = benchmark.run()
    benchmark.export(profiles, save_dir=str(save_dir))

    # ------------------------------------------------------------------
    # 6. Print results table
    # ------------------------------------------------------------------
    _print_results(profiles, save_dir)


def _print_results(profiles: dict, save_dir: Path) -> None:
    """Print a formatted comparison table and save scores.json."""
    print("\n" + "=" * 65)
    print(f"{'Method':<20} {'Truthfulness':>14} {'Informativeness':>16}")
    print("-" * 65)

    summary: dict[str, dict] = {}
    for pipeline_name, runs in profiles.items():
        run_list = runs if isinstance(runs, list) else [runs]
        truth_rates = []
        info_rates = []
        for run in run_list:
            evals = run.get("evaluations", {})
            truth = evals.get("Truthfulness", {})
            info = evals.get("Informativeness", {})
            if "truthfulness_rate" in truth:
                truth_rates.append(truth["truthfulness_rate"])
            if "informativeness_rate" in info:
                info_rates.append(info["informativeness_rate"])

        t_mean = sum(truth_rates) / len(truth_rates) if truth_rates else float("nan")
        i_mean = sum(info_rates) / len(info_rates) if info_rates else float("nan")
        summary[pipeline_name] = {
            "truthfulness_rate": t_mean,
            "informativeness_rate": i_mean,
        }
        print(f"{pipeline_name:<20} {t_mean:>13.1%} {i_mean:>15.1%}")

    print("=" * 65 + "\n")

    out_path = save_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
