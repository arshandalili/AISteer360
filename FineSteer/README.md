# FineSteer

FineSteer Code.



## Overview
we propose FineSteer, a fine-grained activation steering framework for mitigating hallucination in reasoning models. Instead of steering on the whole residual stream or adopting a narrow pool, FineSteer localizes and steers specific internal linear modules, such as attention projections and MLP projections, that are causally associated with hallucinated reasoning behavior.




## Install
Clone our repository and download required libraries.

```
    git clone https://github.com/arshandalili/AISteer360
    cd AISteer360/FineSteer
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
This `venv` virtual environment should be good for both Qwen3.5 and Gemma 4.


## Compute Steer Vectors and Obtain Importance Scores

Run commands below for Qwen3.5. For Gemma 4, files with the same name are in `Gemma 4` folder.
```
    # Compute steering vectors
    python compute_u_ours.py
    
    # Obtain importance scores (via attribution patching)
    python compute_attpatching_ours.py
```
Steering vectors will be saved in `all_u_ours`, and importance scores of each module will be saved in `attpaching_ours`. You may use the code snippet below to inspect the top 3 most important modules.

```
import pickle
import torch


path = "hallucination_attpatch_gemma4_E4B_it.pkl"

with open(path, "rb") as f:
    comp_dict = pickle.load(f)  # {component_name: tensor}

def score(x):
    if isinstance(x, torch.Tensor):
        return x.max().item()  # works for scalar and non-scalar tensors
    return float(x)

top3 = sorted(comp_dict.items(), key=lambda kv: score(kv[1]), reverse=True)[:3]

for i, (name, tensor_val) in enumerate(top3, 1):
    print(f"{i}. {name}: {score(tensor_val):.6f}")
```


## Usage
Once a specific module is selected for steering, run vLLM inference code below for Qwen3.5. For Gemma 4, files with the same name are in `Gemma 4` folder.
```
    # The Qwen inference code is doing steering on layer_3_q_proj.
    python inference.py
```
The inference code is saving outputs on GPQA-Diamond benchmark.