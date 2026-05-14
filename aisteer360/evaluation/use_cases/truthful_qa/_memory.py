from __future__ import annotations

import gc

import torch


def hard_free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def mem_report(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"[mem][{tag}] GPU {i}: alloc={alloc:.2f}G reserved={reserved:.2f}G total={total:.2f}G")
