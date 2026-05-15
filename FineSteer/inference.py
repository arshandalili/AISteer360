"""
vLLM V1 (vLLM>=0.13) Activation Steering on MULTIPLE components via worker hooks.

Supports mixed attention stacks such as Qwen/Qwen3.5-4B:
  - self_attn: q_proj, k_proj, v_proj, o_proj
  - linear_attn: in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj
  - mlp: gate_proj, up_proj, down_proj (plus fused/legacy aliases)

It uses path-based vector passing (driver saves *_vec.pt, worker torch.load).
"""

from __future__ import annotations

import os
import re
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset


# =============================================================================
# Worker extension (runs inside vLLM workers)
# =============================================================================
class ActivationHookWorker:
    def _get_model(self) -> torch.nn.Module:
        if not hasattr(self, "model_runner"):
            raise AttributeError("Worker has no attribute `model_runner` (unexpected for vLLM V1 GPU worker).")
        return self.model_runner.get_model()

    def _ensure_state(self) -> None:
        if not hasattr(self, "_act_hook_handles"):
            self._act_hook_handles = []  # type: ignore[attr-defined]
        if not hasattr(self, "_act_hook_records"):
            self._act_hook_records = {}  # type: ignore[attr-defined]

    def _find_module(self, candidates: List[str]) -> Tuple[str, torch.nn.Module]:
        model = self._get_model()
        modules = dict(model.named_modules())

        # exact match
        for cand in candidates:
            if cand in modules:
                return cand, modules[cand]

        # suffix match (must be unique)
        for cand in candidates:
            suffix_hits = [(n, m) for n, m in modules.items() if n.endswith(cand)]
            if len(suffix_hits) == 1:
                return suffix_hits[0]
            if len(suffix_hits) > 1:
                names = "\n".join(f"  - {n}" for n, _ in suffix_hits[:20])
                raise ValueError(
                    f"Ambiguous suffix match for candidate '{cand}'. "
                    f"Matched {len(suffix_hits)} modules:\n{names}"
                )

        raise ValueError(
            "Could not find target module. Tried candidates:\n" + "\n".join(f"  - {c}" for c in candidates)
        )

    def _mlp_candidates(self, layer_idx: int, kind: str) -> List[str]:
        wrappers = [
            "model.layers.{i}.mlp.{name}",
            "layers.{i}.mlp.{name}",
            "model.model.layers.{i}.mlp.{name}",
        ]
        def wrap(n: str) -> List[str]:
            return [w.format(i=layer_idx, name=n) for w in wrappers]

        # Base HF names
        if kind == "gate_proj":
            cands = wrap("gate_proj")
            # Fused + aliases
            cands += wrap("gate_up_proj")  # fused
            cands += wrap("w1")            # legacy gate
            return cands
        if kind == "up_proj":
            cands = wrap("up_proj")
            cands += wrap("gate_up_proj")  # fused
            cands += wrap("w3")            # legacy up
            return cands
        if kind == "down_proj":
            cands = wrap("down_proj")
            cands += wrap("w2")            # legacy down
            return cands

        # Allow raw name passthrough
        return wrap(kind)

    def _attn_candidates(self, layer_idx: int, kind: str) -> List[str]:
        wrappers = [
            "model.layers.{i}.self_attn.{name}",
            "layers.{i}.self_attn.{name}",
            "model.model.layers.{i}.self_attn.{name}",
        ]
        def wrap(n: str) -> List[str]:
            return [w.format(i=layer_idx, name=n) for w in wrappers]

        # HF names
        if kind in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            cands = wrap(kind)
            # Some stacks fuse qkv:
            if kind in {"q_proj", "k_proj", "v_proj"}:
                cands += wrap("qkv_proj")
                cands += wrap("qkv")  # very loose
            return cands

        return wrap(kind)

    def _linear_attn_candidates(self, layer_idx: int, kind: str) -> List[str]:
        wrappers = [
            "model.layers.{i}.linear_attn.{name}",
            "layers.{i}.linear_attn.{name}",
            "model.model.layers.{i}.linear_attn.{name}",
        ]

        def wrap(n: str) -> List[str]:
            return [w.format(i=layer_idx, name=n) for w in wrappers]

        if kind in {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}:
            return wrap(kind)
        return wrap(kind)

    def _build_candidates(self, layer_idx: int, proj_kind: str) -> Tuple[List[str], str]:
        """Return (candidates, family) where family is 'mlp', 'attn', or 'linear_attn'."""
        if proj_kind in {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"}:
            return self._mlp_candidates(layer_idx, proj_kind), "mlp"
        if proj_kind in {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}:
            return self._attn_candidates(layer_idx, proj_kind), "attn"
        if proj_kind in {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}:
            return self._linear_attn_candidates(layer_idx, proj_kind), "linear_attn"
        # Keep permissive fallback if naming differs.
        return (
            self._mlp_candidates(layer_idx, proj_kind)
            + self._attn_candidates(layer_idx, proj_kind)
            + self._linear_attn_candidates(layer_idx, proj_kind),
            "mixed",
        )

    def install_steering_hook(
        self,
        *,
        layer_idx: int,
        proj_kind: str,
        steering_vector: Any,    # path string recommended
        scale: float,
        fused_part: str = "none",  # 'none' | 'first' | 'middle' | 'second'
        record: bool = False,
        record_max_calls: int = 0,
        record_path: Optional[str] = None,
    ) -> str:
        """Install a forward hook that does out <- out - scale * vec."""
        self._ensure_state()

        # Load vector inside worker (robust)
        if isinstance(steering_vector, str):
            vec_base = torch.load(steering_vector, map_location="cuda").reshape(-1)
        elif torch.is_tensor(steering_vector):
            vec_base = steering_vector.detach().to(device="cuda").reshape(-1)
        elif isinstance(steering_vector, (list, tuple)):
            if len(steering_vector) > 0 and isinstance(steering_vector[0], str):
                steering_vector = [float(x) for x in steering_vector]
            vec_base = torch.tensor(steering_vector, device="cuda").reshape(-1)
        else:
            raise TypeError(f"Unsupported steering_vector type: {type(steering_vector)}")

        candidates, _family = self._build_candidates(layer_idx, proj_kind)
        matched_name, target_module = self._find_module(candidates)

        # If we matched a fused module, infer the requested sub-part when not explicitly set.
        effective_fused_part = fused_part
        is_fused_gate_up = matched_name.endswith("gate_up_proj") or ".gate_up_proj" in matched_name
        is_fused_qkv = ("in_proj_qkv" not in matched_name) and (
            matched_name.endswith("qkv_proj") or ".qkv_proj" in matched_name or matched_name.endswith("qkv")
        )

        if is_fused_gate_up:
            if effective_fused_part == "none":
                if proj_kind == "gate_proj":
                    effective_fused_part = "first"
                elif proj_kind == "up_proj":
                    effective_fused_part = "second"
        elif is_fused_qkv:
            if effective_fused_part == "none":
                if proj_kind == "q_proj":
                    effective_fused_part = "first"
                elif proj_kind == "k_proj":
                    effective_fused_part = "middle"
                elif proj_kind == "v_proj":
                    effective_fused_part = "second"
        else:
            # Never split non-fused projections.
            effective_fused_part = "none"

        # Best-effort TP rank/world detection for per-shard vector alignment.
        tp_rank, tp_world = 0, 1
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                tp_rank = int(dist.get_rank())
                tp_world = int(dist.get_world_size())
        except Exception:
            pass
        if tp_world <= 1:
            tp_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            tp_world = int(os.environ.get("WORLD_SIZE", "1"))
        if tp_world <= 0:
            tp_rank, tp_world = 0, 1

        if record:
            self._act_hook_records.setdefault(matched_name, [])  # type: ignore[attr-defined]

        def _hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any):
            out = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(out):
                return output

            vec = vec_base.to(dtype=out.dtype, device=out.device)
            while vec.ndim < out.ndim:
                vec = vec.unsqueeze(0)

            def _candidate_local_vecs(v: torch.Tensor) -> List[torch.Tensor]:
                cands: List[torch.Tensor] = []
                if tp_world > 1 and v.shape[-1] % tp_world == 0:
                    shard = v.reshape(*v.shape[:-1], tp_world, v.shape[-1] // tp_world).select(-2, tp_rank % tp_world)
                    cands.append(shard)
                cands.append(v)
                # dedup by last-dim size/order
                uniq: List[torch.Tensor] = []
                seen = set()
                for t in cands:
                    d = int(t.shape[-1])
                    if d not in seen:
                        uniq.append(t)
                        seen.add(d)
                return uniq

            def _align_lastdim(v: torch.Tensor, target_dim: int, ctx: str) -> torch.Tensor:
                cur_dim = int(v.shape[-1])
                if cur_dim == target_dim:
                    return v
                if target_dim <= 0 or cur_dim % target_dim != 0:
                    raise ValueError(
                        f"Steering dim mismatch for {matched_name} (requested layer_{layer_idx}_{proj_kind}, {ctx}): "
                        f"output_dim={target_dim} vec_dim={cur_dim} tp_rank={tp_rank} tp_world={tp_world}"
                    )
                chunks = cur_dim // target_dim
                if chunks > 1 and tp_world <= 1:
                    raise ValueError(
                        f"Vector appears unsharded for {matched_name} but TP world is {tp_world}. "
                        f"Need sharded vec_dim={target_dim}, got vec_dim={cur_dim}."
                    )
                return v.reshape(*v.shape[:-1], chunks, target_dim).select(-2, tp_rank % chunks)

            def _subtract_full(base: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                v_aligned = _align_lastdim(v, int(base.shape[-1]), "full")
                return base - scale * v_aligned

            out_steered = out

            if effective_fused_part != "none":
                if is_fused_qkv:
                    out_dim = out.shape[-1]
                    vec_dim = vec.shape[-1]
                    # qkv fused projection can be unequal in GQA models (e.g., q=8192, k=v=1024).
                    if out_dim % 3 == 0 and vec_dim == out_dim // 3:
                        third = out_dim // 3
                        vec_third = _align_lastdim(vec, third, "qkv_equal_third")
                        if effective_fused_part == "first":      # q
                            out_steered = torch.cat(
                                [out[..., :third] - scale * vec_third, out[..., third:]], dim=-1
                            )
                        elif effective_fused_part == "middle":   # k
                            out_steered = torch.cat(
                                [
                                    out[..., :third],
                                    out[..., third : 2 * third] - scale * vec_third,
                                    out[..., 2 * third :],
                                ],
                                dim=-1,
                            )
                        elif effective_fused_part == "second":   # v
                            out_steered = torch.cat(
                                [out[..., : 2 * third], out[..., 2 * third :] - scale * vec_third], dim=-1
                            )
                        else:
                            out_steered = _subtract_full(out, vec)
                    else:
                        # Unequal split fallback:
                        # - first (q): vec_dim = q_dim, remaining split equally to k/v
                        # - middle/second (k/v): vec_dim = kv_dim, q_dim = out_dim - 2*kv_dim
                        if effective_fused_part == "first":
                            selected = None
                            for cand in _candidate_local_vecs(vec):
                                q_dim = int(cand.shape[-1])
                                rem = int(out_dim - q_dim)
                                if q_dim > 0 and rem > 0 and rem % 2 == 0:
                                    selected = cand
                                    break
                            if selected is None:
                                raise ValueError(
                                    f"Cannot infer unequal qkv split for {matched_name}: "
                                    f"out_dim={out_dim}, vec_dim={vec_dim}, part={effective_fused_part}, "
                                    f"tp_rank={tp_rank}, tp_world={tp_world}"
                                )
                            q_dim = int(selected.shape[-1])
                            out_steered = torch.cat([out[..., :q_dim] - scale * selected, out[..., q_dim:]], dim=-1)
                        elif effective_fused_part in {"middle", "second"}:
                            selected = None
                            q_dim = -1
                            for cand in _candidate_local_vecs(vec):
                                kv_dim = int(cand.shape[-1])
                                q_try = int(out_dim - 2 * kv_dim)
                                if q_try >= 0:
                                    selected = cand
                                    q_dim = q_try
                                    break
                            if selected is None:
                                raise ValueError(
                                    f"Invalid qkv dims for {matched_name}: "
                                    f"out_dim={out_dim}, vec_dim={vec_dim}, part={effective_fused_part}, "
                                    f"tp_rank={tp_rank}, tp_world={tp_world}"
                                )
                            kv_dim = int(selected.shape[-1])
                            if effective_fused_part == "middle":
                                out_steered = torch.cat(
                                    [
                                        out[..., :q_dim],
                                        out[..., q_dim : q_dim + kv_dim] - scale * selected,
                                        out[..., q_dim + kv_dim :],
                                    ],
                                    dim=-1,
                                )
                            else:
                                out_steered = torch.cat(
                                    [
                                        out[..., : q_dim + kv_dim],
                                        out[..., q_dim + kv_dim :] - scale * selected,
                                    ],
                                    dim=-1,
                                )
                        else:
                            out_steered = _subtract_full(out, vec)
                elif is_fused_gate_up:
                    # gate_up fused projection: split last dim into 2 equal chunks.
                    if out.shape[-1] % 2 != 0:
                        out_steered = _subtract_full(out, vec)
                    else:
                        half = out.shape[-1] // 2
                        vec_half = _align_lastdim(vec, half, "gate_up_half")
                        if effective_fused_part == "first":
                            out_steered = torch.cat([out[..., :half] - scale * vec_half, out[..., half:]], dim=-1)
                        elif effective_fused_part == "second":
                            out_steered = torch.cat([out[..., :half], out[..., half:] - scale * vec_half], dim=-1)
                        else:
                            out_steered = _subtract_full(out, vec)
                else:
                    out_steered = _subtract_full(out, vec)
            else:
                out_steered = _subtract_full(out, vec)

            if record:
                rec_list = self._act_hook_records[matched_name]  # type: ignore[attr-defined]
                if record_max_calls <= 0 or len(rec_list) < record_max_calls:
                    last = out_steered[-1].detach().float().cpu() if out_steered.ndim >= 2 else out_steered.detach().float().cpu()
                    delta = (out_steered - out).detach().float()
                    rec_list.append({
                        "call_idx": len(rec_list),
                        "shape": tuple(out_steered.shape),
                        "last_token_mean": float(last.mean().item()),
                        "last_token_std": float(last.std().item()),
                        "delta_l2_norm": float(delta.norm().item()),
                        "delta_mean_abs": float(delta.abs().mean().item()),
                        "layer_idx": layer_idx,
                        "proj_kind": proj_kind,
                        "matched_name": matched_name,
                        "fused_part": effective_fused_part,
                        "scale": float(scale),
                    })
                    if record_path is not None:
                        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(record_path, "w") as f:
                            json.dump(self._act_hook_records, f, indent=2)

            if isinstance(output, tuple):
                return (out_steered,) + output[1:]
            return out_steered

        handle = target_module.register_forward_hook(_hook)
        self._act_hook_handles.append(handle)  # type: ignore[attr-defined]

        return (
            f"hooked={matched_name} requested=layer_{layer_idx}_{proj_kind} "
            f"fused_part={effective_fused_part} scale={scale}"
        )

    def remove_all_hooks(self) -> int:
        self._ensure_state()
        n = 0
        for h in list(self._act_hook_handles):  # type: ignore[attr-defined]
            try:
                h.remove()
                n += 1
            except Exception:
                pass
        self._act_hook_handles.clear()  # type: ignore[attr-defined]
        return n

    def dump_hook_records(self, out_path: str) -> str:
        self._ensure_state()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self._act_hook_records, f, indent=2)  # type: ignore[attr-defined]
        return out_path

    def get_hook_record_counts(self) -> Dict[str, int]:
        self._ensure_state()
        return {k: len(v) for k, v in self._act_hook_records.items()}  # type: ignore[attr-defined]


# =============================================================================
# Driver
# =============================================================================
_UKEY_RE = re.compile(r"^layer_(\d+)_(.+)$")

def parse_u_key(u_key: str) -> Tuple[int, str]:
    m = _UKEY_RE.match(u_key)
    if not m:
        raise ValueError(f"Bad u_key format: {u_key} (expected layer_<int>_<name>)")
    return int(m.group(1)), m.group(2)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep_artifacts", action="store_true",
                        help="Keep *_vec.pt and hook_records_*.json files.")
    args = parser.parse_args()

    # -------------------------
    # EDIT THESE
    # -------------------------
    model_name = "Qwen/Qwen3.5-4B"
    # Point this to your fine-grained steering-vector pickle.
    u_vec_path = "all_u_ours/hallucination_u_qwen35_4B.pkl"

    # Your 3 components (example)
    u_keys = [
        "layer_3_q_proj",
    ]

    default_scale = 0.1
    per_key_scale: Dict[str, float] = {
        # "layer_31_up_proj": 1.5,
        # "layer_31_gate_proj": 1.0,
        # "layer_16_q_proj": 0.5,
    }

    out_dir = "output/GPQA/"
    os.makedirs(out_dir, exist_ok=True)

    hook_live_path = os.path.join(out_dir, "hook_records_live.json")
    hook_final_path = os.path.join(out_dir, "hook_records_final.json")
    saved_vec_paths: List[str] = []

    # worker_extension_cls must be importable
    module_name = Path(__file__).stem
    worker_ext = f"{module_name}.ActivationHookWorker"

    import vllm
    print("=" * 80)
    print("vLLM version:", getattr(vllm, "__version__", "unknown"))
    print("worker_extension_cls =", worker_ext)
    print("u_keys =", u_keys)
    print("=" * 80)

    with open(u_vec_path, "rb") as f:
        u_dict = pickle.load(f)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=2,
        worker_extension_cls=worker_ext,
        enforce_eager=True,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.96,
    )

    # Install one hook per component
    hook_results: Dict[str, Any] = {}
    matched_names: Dict[str, str] = {}
    for k in u_keys:
        if k not in u_dict:
            raise KeyError(f"{k} not found in u_dict")

        vec_cpu = u_dict[k].detach().to("cpu").reshape(-1)
        vec_path = os.path.join(out_dir, f"{k}_vec.pt")
        torch.save(vec_cpu, vec_path)
        saved_vec_paths.append(vec_path)

        layer_idx, proj_kind = parse_u_key(k)
        scale = float(per_key_scale.get(k, default_scale))

        # Let worker decide fused-part only after it knows matched module name.
        fused_part = "none"

        res = llm.collective_rpc(
            "install_steering_hook",
            kwargs=dict(
                layer_idx=layer_idx,
                proj_kind=proj_kind,
                steering_vector=vec_path,     # pass PATH via RPC
                scale=scale,
                fused_part=fused_part,
                record=True,
                record_max_calls=50,
                record_path=hook_live_path,
            ),
        )
        hook_results[k] = res
        print(f"[installed] {k} -> {res}")
        res_text = " ".join(str(x) for x in res) if isinstance(res, (list, tuple)) else str(res)
        m = re.search(r"hooked=([^\s]+)", res_text)
        if m:
            matched_names[k] = m.group(1)

    print("All hook installs:", hook_results)
    if matched_names:
        reverse: Dict[str, List[str]] = {}
        for k, name in matched_names.items():
            reverse.setdefault(name, []).append(k)
        collisions = {name: keys for name, keys in reverse.items() if len(keys) > 1}
        if collisions:
            print("[warning] Multiple u_keys mapped to the same module:")
            for name, keys in collisions.items():
                print(f"  {name}: {keys}")
        else:
            print("[info] No module-name collisions detected across u_keys.")

    # Generate
    dataset = load_dataset("fingertap/GPQA-Diamond")["test"]
    total_input = []
    for d in dataset:
        total_input.append("Answer the question by giving the index (the letter) of the actual correct answer.\n" + d['question'] + "\n<think>\n")

    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=32768, seed=42)
    outputs = llm.generate(total_input, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        with open(os.path.join(out_dir, f"{i}.txt"), "w", encoding="utf-8") as the_file:
            the_file.write(generated_text)

    hook_counts = llm.collective_rpc("get_hook_record_counts")
    print("Hook record counts:", hook_counts)
    total_hook_calls = 0
    for item in hook_counts:
        if isinstance(item, dict):
            total_hook_calls += sum(int(v) for v in item.values())
    if total_hook_calls == 0:
        raise RuntimeError(
            "No hook calls were recorded. In vLLM this usually means graph/compile path bypassed Python hooks; "
            "run with enforce_eager=True and enable_chunked_prefill=False."
        )

    dump_paths = llm.collective_rpc("dump_hook_records", kwargs=dict(out_path=hook_final_path))
    print("Dumped hook records:", dump_paths)

    removed = llm.collective_rpc("remove_all_hooks")
    print("Removed hooks:", removed)

    if not args.keep_artifacts:
        for p in saved_vec_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        for p in (hook_live_path, hook_final_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        print("Cleaned up temp artifacts (pass --keep_artifacts to keep them).")


if __name__ == "__main__":
    main()
