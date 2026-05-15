import pickle
import re
import unicodedata
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import torch


U_PATH = "all_u_ours/hallucination_u_qwen35_4B.pkl"
OUT_PATH = "attpaching_ours/hallucination_attpatch_qwen35_4B.pkl"
MODEL_PATH = "Qwen/Qwen3.5-4B"


with open(U_PATH, "rb") as f:
    u_dict = pickle.load(f)


all_responses = []
for i in range(100):
    file_path = "original_qwen35_4B/" + str(i) + ".txt"
    with open(file_path, "r", encoding="utf-8") as file:
        all_responses.append(file.read())

total_input_all, i = [], 0
with open("instances100_for_u_final.txt", "r", encoding="utf-8") as f:
    for line in f:
        json_object = json.loads(line.strip())
        file_str = json_object["prompt"] + all_responses[i]
        total_input_all.append(file_str)
        i += 1


model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
model.eval()

acts = {}


def make_hook(name):
    def hook(module, input, output):
        # store the activation *with* grad
        layer_output = output[0] if isinstance(output, tuple) else output
        acts[name] = layer_output
        layer_output.retain_grad()
        return None

    return hook


handles, self_attn_layer = [], [3, 7, 11, 15, 19, 23, 27, 31]
for layer_idx, layer in enumerate(model.model.layers):
    if layer_idx in self_attn_layer:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, proj)
            name = f"layer_{layer_idx}_{proj}"
            handles.append(module.register_forward_hook(make_hook(name)))
    else:
        for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            module = getattr(layer.linear_attn, proj)
            name = f"layer_{layer_idx}_{proj}"
            handles.append(module.register_forward_hook(make_hook(name)))

    for proj in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(layer.mlp, proj)
        name = f"layer_{layer_idx}_{proj}"
        handles.append(module.register_forward_hook(make_hook(name)))


count, total = 0, {}
all_modules1 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
all_modules2 = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]
for layer in range(32):
    if layer in self_attn_layer:
        for m in all_modules1:
            l_m = "layer_" + str(layer) + "_" + m
            total[l_m] = 0
    else:
        for m in all_modules2:
            l_m = "layer_" + str(layer) + "_" + m
            total[l_m] = 0

missing_u_keys = set(total.keys()) - set(u_dict.keys())
if missing_u_keys:
    raise KeyError(f"Missing fine-grained steering vectors in u_dict: {sorted(missing_u_keys)}")


##### main loop #######
for j in range(100):
    print(j)
    original = total_input_all[j]

    with open(
        "original_qwen35_4B_annotated/" + str(j) + ".txt",
        "r",
        encoding="utf-8",
    ) as f:
        annotated = f.read()

    labels = ["hallucination"]
    seg_re = re.compile(r'\["(' + "|".join(labels) + r')"\]\s*(.*?)\s*\["end-section"\]', re.DOTALL)
    segments = [{"label": m.group(1), "text": m.group(2).strip()} for m in seg_re.finditer(annotated)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    enc = tokenizer(
        original,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    offsets = enc.offset_mapping
    threshold = 0.99

    flag, target_indices = False, []
    for seg in segments:
        raw = seg["text"]
        sm = difflib.SequenceMatcher(None, original, raw)
        match = max(sm.get_matching_blocks(), key=lambda b: b.size)

        coverage = match.size / len(raw)
        if coverage < threshold:
            continue

        start_char = match.a
        end_char = start_char + match.size

        start_tok = next(
            i for i, (s, e) in enumerate(offsets) if s <= start_char < e
        )
        end_tok = next(
            i for i, (s, e) in enumerate(offsets) if s < end_char <= e
        )
        # Make end index exclusive for downstream slicing.
        end_tok += 1

        if seg["label"] == "hallucination":
            flag = True
            target_indices.append([start_tok, end_tok])

    if flag:
        for indices in target_indices:
            # CausalLM loss is shifted by 1 token: token 0 has no prediction.
            label_start = max(indices[0], 1)
            label_end = indices[1]  # exclusive
            if label_end <= label_start:
                continue

            tokenized_input = tokenizer(original, return_tensors="pt", return_offsets_mapping=True)
            input_ids = tokenized_input["input_ids"][:, :label_end].to(model.device)
            attention_mask = tokenized_input["attention_mask"][:, :label_end].to(model.device)

            labels = input_ids.clone()
            labels[0, :label_start] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if torch.isnan(loss):
                continue

            model.zero_grad()
            loss.backward()
            count += 1

            for name, act in acts.items():
                # Align to shifted CE loss: labels [s:e) <- logits/acts [s-1:e-1)
                grad = act.grad.squeeze(0)[label_start - 1:label_end - 1, :]
                if grad.numel() == 0:
                    continue
                grad_mean = torch.mean(grad, 0)
                total[name] += torch.abs(torch.dot(u_dict[name].to(grad_mean.device), grad_mean.float()))

            del input_ids
            del attention_mask


if count == 0:
    raise ValueError("No valid hallucination spans for gradient attribution.")

for layer in range(32):
    if layer in self_attn_layer:
        for m in all_modules1:
            l_m = "layer_" + str(layer) + "_" + m
            total[l_m] = (total[l_m] / count).cpu()
    else:
        for m in all_modules2:
            l_m = "layer_" + str(layer) + "_" + m
            total[l_m] = (total[l_m] / count).cpu()

with open(OUT_PATH, "wb") as f:
    pickle.dump(total, f)

for h in handles:
    h.remove()
