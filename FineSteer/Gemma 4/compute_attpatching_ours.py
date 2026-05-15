import pickle
import re
import unicodedata
import difflib
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import torch


U_PATH = "all_u_ours/hallucination_u_gemma4_E4B_it.pkl"
OUT_PATH = "attpaching_ours/hallucination_attpatch_gemma4_E4B_it.pkl"
MODEL_PATH = "google/gemma-4-E4B-it"


with open(U_PATH, "rb") as f:
    u_dict = pickle.load(f)


all_responses = []
for i in range(100):
    file_path = "original_gemma4_E4B_it/" + str(i) + ".txt"
    with open(file_path, "r", encoding="utf-8") as file:
        all_responses.append(file.read())

total_input_all, i = [], 0
with open("../instances100_for_u_final.txt", "r", encoding="utf-8") as f:
    for line in f:
        json_object = json.loads(line.strip())
        total_input_all.append(
            {
                "prompt": json_object["prompt"],
                "response": all_responses[i],
                "original": json_object["prompt"] + all_responses[i],
            }
        )
        i += 1


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)
model.eval()
device = next(model.parameters()).device


def get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
    model_layers = model.model.language_model.layers
elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
    model_layers = model.language_model.layers
elif hasattr(model, "model") and hasattr(model.model, "layers"):
    model_layers = model.model.layers
else:
    raise ValueError("Unable to locate transformer layers in loaded model.")

acts = {}


def make_hook(name):
    def hook(module, input, output):
        # store the activation *with* grad
        layer_output = output[0] if isinstance(output, tuple) else output
        acts[name] = layer_output
        layer_output.retain_grad()
        return None

    return hook


target_modules = [
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
    ("per_layer_projection", "per_layer_projection"),
    ("per_layer_input_gate", "per_layer_input_gate"),
]

handles = []
layer_modules = {}
for layer_idx, layer in enumerate(model_layers):
    layer_modules[layer_idx] = []
    for attr_path, key in target_modules:
        module = get_nested_attr(layer, attr_path)
        if module is None:
            continue
        name = f"layer_{layer_idx}_{key}"
        handles.append(module.register_forward_hook(make_hook(name)))
        layer_modules[layer_idx].append((attr_path, key))


count, total = 0, {}
num_layers = len(model_layers)
for layer_idx in range(num_layers):
    for _, key in layer_modules[layer_idx]:
        l_m = "layer_" + str(layer_idx) + "_" + key
        total[l_m] = torch.tensor(0.0)

missing_u_keys = set(total.keys()) - set(u_dict.keys())
if missing_u_keys:
    raise KeyError(f"Missing fine-grained steering vectors in u_dict: {sorted(missing_u_keys)}")


##### main loop #######
for j in range(100):
    print(j)
    prompt = total_input_all[j]["prompt"]
    response = total_input_all[j]["response"]
    original = total_input_all[j]["original"]
    chat_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    with open(
        "original_gemma4_E4B_it_annotated/" + str(j) + ".txt",
        "r",
        encoding="utf-8",
    ) as f:
        annotated = f.read()

    labels = ["hallucination"]
    seg_re = re.compile(r'\["(' + "|".join(labels) + r')"\]\s*(.*?)\s*\["end-section"\]', re.DOTALL)
    segments = [{"label": m.group(1), "text": m.group(2).strip(), "start": m.start()} for m in seg_re.finditer(annotated)]
    doc_start = len(annotated) - len(annotated.lstrip())

    enc = tokenizer(
        chat_text,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )
    offsets = enc.offset_mapping
    threshold = 0.99

    prompt_start_in_chat = chat_text.find(prompt)
    response_start_in_chat = chat_text.rfind(response)
    if prompt_start_in_chat == -1 or response_start_in_chat == -1:
        print("chat template text mapping failed!!!")
        break

    flag, target_indices = False, []
    skipped_first_start_hallucination = False
    for seg_idx, seg in enumerate(segments):
        raw = seg["text"]
        if not raw:
            continue

        sm = difflib.SequenceMatcher(None, original, raw)
        match = max(sm.get_matching_blocks(), key=lambda b: b.size)

        coverage = match.size / len(raw)
        if coverage < threshold:
            continue

        start_char = match.a
        end_char = start_char + match.size

        if start_char < len(prompt):
            start_char_chat = prompt_start_in_chat + start_char
        else:
            start_char_chat = response_start_in_chat + (start_char - len(prompt))

        if end_char <= len(prompt):
            end_char_chat = prompt_start_in_chat + end_char
        else:
            end_char_chat = response_start_in_chat + (end_char - len(prompt))

        start_tok = next(
            i for i, (s, e) in enumerate(offsets) if s <= start_char_chat < e
        )
        end_tok = next(
            i for i, (s, e) in enumerate(offsets) if s < end_char_chat <= e
        )
        # Make end index exclusive for downstream slicing.
        end_tok += 1

        if seg["label"] == "hallucination":
            if (
                not skipped_first_start_hallucination
                and seg_idx == 0
                and seg["start"] == doc_start
            ):
                skipped_first_start_hallucination = True
                continue
            flag = True
            target_indices.append([start_tok, end_tok])

    if flag:
        for indices in target_indices:
            # CausalLM loss is shifted by 1 token: token 0 has no prediction.
            label_start = max(indices[0], 1)
            label_end = indices[1]  # exclusive
            if label_end <= label_start:
                continue

            tokenized_input = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized_input["input_ids"][:, :label_end].to(device)
            attention_mask = tokenized_input["attention_mask"][:, :label_end].to(device)

            labels = input_ids.clone()
            labels[0, :label_start] = -100

            acts.clear()
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
                score = torch.abs(torch.dot(u_dict[name].to(device=grad_mean.device, dtype=grad_mean.dtype), grad_mean,))
                total[name] += score.detach().float().cpu()

            del input_ids
            del attention_mask


if count == 0:
    raise ValueError("No valid hallucination spans for gradient attribution.")

for name in total:
    total[name] = (total[name] / count).cpu()

with open(OUT_PATH, "wb") as f:
    pickle.dump(total, f)

for h in handles:
    h.remove()
