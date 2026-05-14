"""Per-method, per-prompt diff between AI and SRC JSONLs.

Looks for systematic patterns: is AI consistently shorter? More refusals?
Random differences (= GPU noise) vs systematic (= bug)?
"""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path


AI_DIR = "results/raw/Llama3.1-8B-Base"
SRC_DIR = "/storage/work/sbd5760/odesteer/results/truthfulqa/raw_outputs/Llama3.1-8B-Base"


def find_src(method_label: str) -> str | None:
    for f in os.listdir(SRC_DIR):
        if f.startswith(f"Llama3.1-8B-Base-l13-{method_label}-") and f.endswith("seed42.jsonl"):
            return os.path.join(SRC_DIR, f)
    return None


PAIRS = [
    ("RepE-T1.0", "repe-T1.0"),
    ("CAA-T5.0", "caa-T5.0"),
    ("ITI-T1.0", "iti-T1.0"),
    ("MiMiC-T1.0", "mimic-T1.0"),
    ("LinAcT-T1.0", "lin_act-T1.0"),
]


def analyze(label_src: str, label_ai: str) -> None:
    ai_path = os.path.join(AI_DIR, f"l13-{label_ai}-seed42.jsonl")
    src_path = find_src(label_src)
    if not src_path or not os.path.exists(ai_path):
        print(f"[skip] {label_src} (ai={Path(ai_path).exists()}, src={src_path is not None})")
        return

    ai = {r["prompt"]: r["output"] for r in (json.loads(l) for l in open(ai_path))}
    src = {r["prompt"]: r["output"] for r in (json.loads(l) for l in open(src_path))}
    common = sorted(set(ai) & set(src))
    n = len(common)

    ai_lens = [len(ai[q]) for q in common]
    src_lens = [len(src[q]) for q in common]
    exact = sum(1 for q in common if ai[q] == src[q])
    pref20 = sum(1 for q in common if ai[q][:20] == src[q][:20])

    ai_refusal = sum(1 for q in common if "no comment" in ai[q].lower() or "i don't know" in ai[q].lower() or "not sure" in ai[q].lower())
    src_refusal = sum(1 for q in common if "no comment" in src[q].lower() or "i don't know" in src[q].lower() or "not sure" in src[q].lower())

    ai_longer = sum(1 for q in common if len(ai[q]) > len(src[q]))
    src_longer = sum(1 for q in common if len(src[q]) > len(ai[q]))

    print(f"\n========= {label_src} =========")
    print(f"  n_common={n}, exact={exact} ({exact/n:.1%}), prefix-20 match={pref20} ({pref20/n:.1%})")
    print(f"  char-len  AI mean={statistics.mean(ai_lens):6.1f} median={statistics.median(ai_lens):6.0f} | "
          f"SRC mean={statistics.mean(src_lens):6.1f} median={statistics.median(src_lens):6.0f}")
    print(f"  AI-longer={ai_longer} ({ai_longer/n:.1%})  SRC-longer={src_longer} ({src_longer/n:.1%})")
    print(f"  refusals  AI={ai_refusal} ({ai_refusal/n:.1%})  SRC={src_refusal} ({src_refusal/n:.1%})")

    # Show 2 representative diffs
    diffs = [q for q in common if ai[q] != src[q]]
    print(f"  example diffs (2 of {len(diffs)}):")
    for q in diffs[:2]:
        print(f"    Q: {q[:70]}")
        print(f"      AI ({len(ai[q]):>3}): {ai[q][:90]!r}")
        print(f"      SRC({len(src[q]):>3}): {src[q][:90]!r}")


def main():
    print(f"AI   dir: {AI_DIR}")
    print(f"SRC  dir: {SRC_DIR}")
    for src_label, ai_label in PAIRS:
        analyze(src_label, ai_label)


if __name__ == "__main__":
    main()
