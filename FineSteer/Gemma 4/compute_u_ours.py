import pickle
import re
import unicodedata
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import torch


model_name = "google/gemma-4-E4B-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


def get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def make_hook(layer_idx: int, proj_name: str):
    def hook(module, inputs, output):
        # Save the full-sequence output of this linear layer
        activations[f"layer_{layer_idx}_{proj_name}"] = torch.squeeze(output.detach())
    return hook

if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
    model_layers = model.model.language_model.layers
elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
    model_layers = model.language_model.layers
elif hasattr(model, "model") and hasattr(model.model, "layers"):
    model_layers = model.model.layers
else:
    raise ValueError("Unable to locate transformer layers in loaded model.")

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

layer_modules = {}
for layer_idx, layer in enumerate(model_layers):
    layer_modules[layer_idx] = []
    for attr_path, key in target_modules:
        lin_mod = get_nested_attr(layer, attr_path)
        if lin_mod is None:
            continue
        lin_mod.register_forward_hook(make_hook(layer_idx, key))
        layer_modules[layer_idx].append((attr_path, key))


all_responses = []
for i in range(100):
    file_path = "original_gemma4_E4B_it/" + str(i) + ".txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        all_responses.append(file.read())

total_input_all, i = [], 0
with open('../instances100_for_u_final.txt', 'r', encoding='utf-8') as f:
    for line in f:
        json_object = json.loads(line.strip())
        total_input_all.append({
            "prompt": json_object['prompt'],
            "response": all_responses[i],
            "original": json_object['prompt'] + all_responses[i],
        })
        i += 1
print("len(total_input_all):", len(total_input_all))


hallucination = ["hallucination"]
D_plus_act, D_minus_act, D_plus_count = {}, {}, [0]
total_sum = {}
device = next(model.parameters()).device
num_layers = len(model_layers)
for layer_idx, layer in enumerate(model_layers):
    for attr_path, key in layer_modules[layer_idx]:
        lin_mod = get_nested_attr(layer, attr_path)
        l_m = "layer_" + str(layer_idx) + "_" + key
        out_dim = lin_mod.out_features
        D_plus_act[l_m] = [torch.zeros(out_dim, device=device)]
        D_minus_act[l_m] = torch.zeros(out_dim, device=device)
        total_sum[l_m] = torch.zeros(out_dim, device=device)

reference_key = None
for layer_idx in range(num_layers):
    if layer_modules[layer_idx]:
        reference_key = "layer_" + str(layer_idx) + "_" + layer_modules[layer_idx][0][1]
        break
if reference_key is None:
    raise ValueError("No target linear modules found for steering.")

total_len = 0

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

    with open("original_gemma4_E4B_it_annotated/" + str(j) + ".txt", "r", encoding="utf-8") as f:
        annotated = f.read()

    activations = {}
    inputs = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    _ = model(input_ids, use_cache=False)

    total_len += activations[reference_key].size(0)
    for layer_idx in range(num_layers):
        for _, key in layer_modules[layer_idx]:
            l_m = "layer_" + str(layer_idx) + "_" + key
            D_minus_act[l_m] += torch.mean(activations[l_m], 0).to(device)
            total_sum[l_m] += activations[l_m].sum(dim=0).to(device)



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
    if len(enc['input_ids']) != activations[reference_key].size()[0]:
        print("sequence dimension mismatch!!!")
        break

    prompt_start_in_chat = chat_text.find(prompt)
    response_start_in_chat = chat_text.rfind(response)
    if prompt_start_in_chat == -1 or response_start_in_chat == -1:
        print("chat template text mapping failed!!!")
        break

    threshold = 0.99

    # -----------------------------------------------------------------------------
    # hallucination loop
    # -----------------------------------------------------------------------------
    for c in range(1):
        print(hallucination[c])
        flag, target_indices = False, []
        skipped_first_start_hallucination = False
        for seg_idx, seg in enumerate(segments):
            raw = seg["text"]
            sm  = difflib.SequenceMatcher(None, original, raw)
            # find the longest matching contiguous block
            match = max(sm.get_matching_blocks(), key=lambda b: b.size)

            # how much of the snippet did we cover?
            coverage = match.size / len(raw)
            if coverage < threshold:
                continue
                raise ValueError(
                    f"Snippet {seg['label']!r} only {coverage:.0%} matched (< {threshold:.0%} required):\n{raw!r}"
                )

            # character span in the ORIGINAL string
            start_char = match.a
            end_char   = start_char + match.size

            if start_char < len(prompt):
                start_char_chat = prompt_start_in_chat + start_char
            else:
                start_char_chat = response_start_in_chat + (start_char - len(prompt))

            if end_char <= len(prompt):
                end_char_chat = prompt_start_in_chat + end_char
            else:
                end_char_chat = response_start_in_chat + (end_char - len(prompt))

            # map to token indices
            start_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s <= start_char_chat < e
            )
            end_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s < end_char_chat <= e
            )

            if seg['label'] == hallucination[c]:
                if (
                    not skipped_first_start_hallucination
                    and seg_idx == 0
                    and seg["start"] == doc_start
                ):
                    skipped_first_start_hallucination = True
                    continue
                flag = True
                target_indices.append([start_tok, end_tok])
                # print(original[start_char: end_char])
        if flag:
            D_plus_count[c] += 1
            rows = []
            for indices in target_indices:
                rows += list(range(max(0, indices[0] - 5), indices[1]))
            for layer_idx in range(num_layers):
                for _, key in layer_modules[layer_idx]:
                    l_m = "layer_" + str(layer_idx) + "_" + key
                    D_plus_act[l_m][c] += torch.mean(activations[l_m][rows, :], 0).to(device)


    del activations

names = ["hallucination"]
for c in range(1):
    u = {}
    for layer_idx in range(num_layers):
        for _, key in layer_modules[layer_idx]:
            l_m = "layer_" + str(layer_idx) + "_" + key
            temp_u = D_plus_act[l_m][c] / D_plus_count[c] - D_minus_act[l_m] / len(total_input_all)
            a_overall_mean = total_sum[l_m] / total_len
            u[l_m] = temp_u * (torch.norm(a_overall_mean, p=2) / torch.norm(temp_u, p=2))
            u[l_m] = u[l_m].cpu()


    with open('all_u_ours/' + names[c] + '_u_gemma4_E4B_it.pkl', 'wb') as f:
        pickle.dump(u, f)
