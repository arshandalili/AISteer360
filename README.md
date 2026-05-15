![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-light-mode-only)

[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/AISteer360/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://img.shields.io/github/license/generative-computing/mellea)

---

# TQABench: TruthfulQA steering benchmark

The `TQABench` branch is a reproducible TruthfulQA pipeline for activation steering, ported from the [ODESteer source repo](https://github.com/arshandalili/odesteer). It registers 11 single-layer steering methods as AISteer360 state controls, runs the source repo's 2-fold cross-validation generation pipeline, and evaluates with the original AllenAI judges.

## Baselines

All methods live under `aisteer360/algorithms/state_control/<method>/` and share an `Args` of `{layer_id, T}` (plus method-specific kwargs where applicable).

| Method | Type | Source paper |
|:---|:---|:---|
| `repe` | PCA-on-difference unit vector | [Representation Engineering (Zou et al., 2023)](https://arxiv.org/abs/2310.01405) |
| `caa` *(override)* | Mean-difference steering vector | [CAA (Panickssery et al., 2023)](https://arxiv.org/abs/2312.06681) |
| `iti` *(override)* | Per-layer logistic-regression direction | [ITI (Li et al., 2023)](https://arxiv.org/abs/2306.03341) |
| `mimic` | Affine optimal-transport steering | [MiMiC (Ravfogel et al., 2024)](https://openreview.net/forum?id=GwA4go0Mw4) |
| `lin_act` | Per-dimension linear OT | [LinAcT (Rodriguez et al., 2025)](https://openreview.net/forum?id=l2zFn6TIQi) |
| `sphere_steer` *(override)* | vMF + SLERP rotation to truthful prototype | [Spherical Steering](https://arxiv.org/pdf/2602.08169) |
| `cobras` | Sinkhorn-regularized OT on the hypersphere | ODESteer paper |
| `ode_steer` | Kernel-gradient ODE (NormedPoly classifier) | ODESteer paper |
| `rff_ode_steer` | Kernel-gradient ODE (RFF classifier) | ODESteer paper |
| `step_ode_steer` | Single-Euler-step ODE (NormedPoly) | ODESteer paper |
| `rff_step_ode_steer` | Single-Euler-step ODE (RFF) | ODESteer paper |

The three overrides replace AISteer360's prior implementations with the source repo's math: a single-layer last-token hook with external pos/neg activations passed through `pipeline.steer(...)`.

## Install

```bash
git clone -b TQABench https://github.com/arshandalili/AISteer360.git
cd AISteer360
uv venv .venv --python 3.11
uv pip install -e .
```

Pre-computed positive/negative activations live in `data/truthfulqa/activations/<model_name>/`:

```
data/truthfulqa/
├── activations/Llama3.1-8B-Base/{pos,neg}_{0,1}_activations_layer13.pt
└── texts/{pos,neg}_{0,1}.jsonl
```

## Running the benchmark

Single method:

```bash
.venv/bin/python scripts/reproduce_truthfulqa.py \
    --model meta-llama/Llama-3.1-8B \
    --data-model-name Llama3.1-8B-Base \
    --layer 13 --method caa --T 5.0 \
    --seed 42 --batch-size 8 --dtype float32 --judge-dtype auto
```

Per-prompt JSONL goes to `results/raw/Llama3.1-8B-Base/l13-caa-T5.0-seed42.jsonl`. One row is appended to `results/eval/Llama3.1-8B-Base/l13-TruthfulQA-seed42.csv` with columns `Model, Steering Method, True * Info, Truthfulness, Informativeness, Perplexity, Dist-1, Dist-2, Dist-3, n_samples`.

Sweep all 11 baselines:

```bash
./scripts/run_full_sweep.sh
```

Override defaults via env: `MODEL=… DATA_MODEL_NAME=… LAYER=… SEED=… BATCH=… ./scripts/run_full_sweep.sh`.

## Differences from `main`

- **Overrides** `aisteer360/algorithms/state_control/{caa,iti,sphere_steer}/` with source-repo math. Their `Args` no longer accept `data`/`steering_vector`/`multiplier`/etc. — stay on `main` if you need the original implementations.
- **Adds** eight state controls under `aisteer360/algorithms/state_control/{repe,mimic,lin_act,cobras,ode_steer,rff_ode_steer,step_ode_steer,rff_step_ode_steer}/`.
- **Adds** AllenAI Truthfulness/Informativeness judges, GPT2-XL perplexity, and Distinct-N under `aisteer360/evaluation/metrics/custom/truthful_qa/`. The existing Qwen-based `Truthfulness` and `Informativeness` are unchanged.
- **Rewrites** `aisteer360/evaluation/use_cases/truthful_qa/use_case.py` for 2-fold CV; the constructor now takes `model_name` and `layer_idx`.
- **Pins** `transformers==4.52.x`, `torch==2.6.x`, `accelerate==1.7.0`, `numpy<2`, `scikit-learn<1.8`, and adds `pot`, `torchdiffeq`, `lightning`. Stricter than `main` to match the source stack for byte-equivalent generation.

---

# Other methods and data

The rest of this README is the upstream AISteer360 documentation: the toolkit's overview, install, featured applications, and full control library.

## About AISteer360

The AI Steerability 360 toolkit is a library for general-purpose steering of LLMs. It supports steering methods across input, structural, state, and output control surfaces, composes them into a `SteeringPipeline`, and compares methods (and pipelines) on custom tasks and metrics.

See the [documentation](https://ibm.github.io/AISteer360/) to get started.

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) (Python 3.11+):

```commandline
uv venv --python 3.11 && uv pip install .
```

Activate with `source .venv/bin/activate`. On Windows, split into two commands instead of chaining with `&&`.

Inference runs through Hugging Face. Create a `.env` file with:

```
HUGGINGFACE_TOKEN=hf_***
```

Some models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are gated — make sure your token's account has access.

> [!NOTE]
> AISteer360 runs the model in-process. Use a machine with enough GPU memory for the base checkpoint plus whatever overhead your steering method adds.

## Featured applications

Benchmarking and comparing steering methods on realistic tasks is a core feature. The examples below show this in practice.

| <div style="font-weight: bold; text-align: left;">Steering for instruction following</div> |
|:---|
| Studies how post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)) affects instruction following. Sweeps steering strength and looks at the trade-off between instruction following and general response quality.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/benchmark_instruction_following/instruction_following.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-weight: bold; text-align: left;">Steering for commonsense reasoning</div> |
|:---|
| Benchmarks steering methods on [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa), comparing few-shot prompting against a LoRA adapter trained with DPO. Sweeps the number of few-shot examples across two models and compares to the fine-tuned baseline.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/benchmark_commonsense_mcqa/commonsense_mcqa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-weight: bold; text-align: left;">Composite steering for truthfulness</div> |
|:---|
| Composes a state control ([PASTA](https://arxiv.org/abs/2311.02262)) with an output control ([DeAL](https://arxiv.org/abs/2402.06147)) to improve truthfulness on [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA) without sacrificing informativeness. Sweeps the joint parameter space and compares each control to the composition.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/benchmark_truthful_qa_composite_steering/truthful_qa_composite_steering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Control library

Each method has a demo notebook under `examples/notebooks/control_*`.

| Method | Authors | Notebook |
|:-------|:----------|:---------|
| [Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248) | Turner et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_act_add/act_add.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681) | Panickssery et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_caa/caa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Conditional Activation Steering (CAST)](https://arxiv.org/abs/2409.05907) | Lee et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_cast/cast.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Decoding-time Alignment (DeAL)](https://arxiv.org/abs/2402.06147) | Huang et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_deal/deal.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Few-shot Learning](https://arxiv.org/abs/2005.14165) | Brown et al., 2020 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_few_shot/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341) | Li et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_iti/iti.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Post-hoc Attention Steering (PASTA)](https://arxiv.org/abs/2311.02262) | Zhang et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_pasta/pasta.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Reward-Augmented Decoding (RAD)](https://arxiv.org/abs/2310.09520) | Deng & Raffel, 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_rad/rad.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Self-Disciplined Autoregressive Sampling (SASA)](https://arxiv.org/abs/2410.03818) | Ko et al., 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_sasa/sasa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Thinking Intervention](https://arxiv.org/abs/2503.24370) | Wu et al., 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/control_thinking_intervention/thinking_intervention.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

Wrappers for external libraries:

| Wrapper | Authors | Notebook |
|:--------|:----------|:---------|
| [MergeKit](https://github.com/arcee-ai/mergekit) | Goddard et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/wrapper_mergekit/mergekit_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [TRL](https://github.com/huggingface/trl) | von Werra et al., 2020 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/wrapper_trl/trl_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Contributing

Contributions are welcome — new steering methods (controls), new evaluations (use cases, metrics), bug reports, docs, or features. See the [contribution guidelines](CONTRIBUTING.md) for the basics, and the tutorials below for specifics.

- [Adding a new steering method](./docs/tutorials/add_new_steering_method.md)
- [Adding a new use case / benchmark](./docs/tutorials/add_new_use_case.md) — see `aisteer360/evaluation/use_cases/` and `aisteer360/evaluation/benchmark.py`
- [Adding a new metric](./docs/tutorials/add_new_metric.md) — generic metrics live in `aisteer360/evaluation/metrics/`

## Reference

```bibtex
@article{miehling2026aisteerability360,
  title = {AI Steerability 360: A Toolkit for Steering Large Language Models},
  author = {Miehling, Erik and Ramamurthy, Karthikeyan Natesan and Venkateswaran, Praveen and Ko, Irene and Dognin, Pierre and Singh, Moninder and Pedapati, Tejaswini and Balakrishnan, Avinash and Riemer, Matthew and Wei, Dennis and Vejsbjerg, Inge and Daly, Elizabeth M. and Varshney, Kush R.},
  journal = {arXiv preprint arXiv:2603.07837},
  year = {2026}
}
```

## IBM ❤️ Open Source AI

The AI Steerability 360 toolkit has been brought to you by IBM.
