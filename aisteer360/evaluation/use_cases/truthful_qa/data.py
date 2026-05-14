from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor


_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "truthfulqa"


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR


def load_tqa_gen_questions(split_idx: int, data_dir: str | Path | None = None) -> list[str]:
    base = get_data_dir(data_dir) / "texts"
    df = pd.read_json(base / f"pos_{split_idx}.jsonl", lines=True, orient="records")
    return df["question"].unique().tolist()


def load_tqa_gen_data(
    model_name: str,
    layer_idx: int,
    split_idx: int,
    data_dir: str | Path | None = None,
) -> tuple[Tensor, Tensor]:
    base = get_data_dir(data_dir) / "activations" / model_name
    pos = torch.load(
        base / f"pos_{split_idx}_activations_layer{layer_idx}.pt",
        weights_only=True,
        map_location="cpu",
    )
    neg = torch.load(
        base / f"neg_{split_idx}_activations_layer{layer_idx}.pt",
        weights_only=True,
        map_location="cpu",
    )
    return pos, neg


def load_tqa_answers(
    split_idx: int,
    data_dir: str | Path | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    base = get_data_dir(data_dir) / "texts"
    pos_df = pd.read_json(base / f"pos_{split_idx}.jsonl", lines=True, orient="records")
    neg_df = pd.read_json(base / f"neg_{split_idx}.jsonl", lines=True, orient="records")
    correct: dict[str, list[str]] = defaultdict(list)
    incorrect: dict[str, list[str]] = defaultdict(list)
    for _, row in pos_df.iterrows():
        correct[row["question"]].append(row["answer"])
    for _, row in neg_df.iterrows():
        incorrect[row["question"]].append(row["answer"])
    return dict(correct), dict(incorrect)


def build_evaluation_data(
    splits: tuple[int, ...] = (0, 1),
    data_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for split_idx in splits:
        questions = load_tqa_gen_questions(split_idx, data_dir)
        correct, incorrect = load_tqa_answers(split_idx, data_dir)
        for q in questions:
            items.append(
                {
                    "question": q,
                    "split": split_idx,
                    "correct_answers": correct.get(q, []),
                    "incorrect_answers": incorrect.get(q, []),
                }
            )
    return items
