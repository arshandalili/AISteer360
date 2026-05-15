import pickle
import re
import unicodedata
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import torch


model_name = "Qwen/Qwen3.5-4B"
tokenizer  = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B')
model      = AutoModelForCausalLM.from_pretrained(model_name)


def make_hook(layer_idx: int, proj_name: str):
    def hook(module, inputs, output):
        # Save the full-sequence output of this linear layer
        activations[f"layer_{layer_idx}_{proj_name}"] = torch.squeeze(output.detach())
    return hook

self_attn_layer = [3, 7, 11, 15, 19, 23, 27, 31]
for layer_idx, layer in enumerate(model.model.layers):
    if layer_idx in self_attn_layer:
        # Self-attention projections
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin_mod = getattr(layer.self_attn, proj)
            lin_mod.register_forward_hook(make_hook(layer_idx, proj))
    else:
        # Linear-attention projections
        for proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            lin_mod = getattr(layer.linear_attn, proj)
            lin_mod.register_forward_hook(make_hook(layer_idx, proj))

    # MLP projections
    for proj in ("gate_proj", "up_proj", "down_proj"):
        lin_mod = getattr(layer.mlp, proj)
        lin_mod.register_forward_hook(make_hook(layer_idx, proj))


all_responses = []
for i in range(100):
    file_path = "original_qwen35_4B/" + str(i) + ".txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        all_responses.append(file.read())

total_input_all, i = [], 0
with open('instances100_for_u_final.txt', 'r', encoding='utf-8') as f:
    for line in f:
        json_object = json.loads(line.strip())
        file_str = json_object['prompt'] + all_responses[i]
        total_input_all.append(file_str)
        i += 1
print("len(total_input_all):", len(total_input_all))


hallucination = ["hallucination"]
all_modules1, D_plus_act, D_minus_act, D_plus_count = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], {}, {}, [0]
all_modules2 = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "gate_proj", "up_proj", "down_proj"]
total_sum = {}
for layer in range(32):
    if layer in self_attn_layer:
        for m in all_modules1:
            l_m = "layer_" + str(layer) + "_" + m
            if m in ["o_proj", "down_proj"]:
                D_plus_act[l_m] = [torch.zeros(2560, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(2560, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(2560, device=torch.device('cuda'))
            elif m in ["k_proj", "v_proj"]:
                D_plus_act[l_m] = [torch.zeros(1024, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(1024, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(1024, device=torch.device('cuda'))
            elif m in ["q_proj"]:
                D_plus_act[l_m] = [torch.zeros(8192, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(8192, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(8192, device=torch.device('cuda'))
            elif m in ["gate_proj", "up_proj"]:
                D_plus_act[l_m] = [torch.zeros(9216, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(9216, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(9216, device=torch.device('cuda'))
    else:
        for m in all_modules2:
            l_m = "layer_" + str(layer) + "_" + m
            if m in ["out_proj", "down_proj"]:
                D_plus_act[l_m] = [torch.zeros(2560, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(2560, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(2560, device=torch.device('cuda'))
            elif m in ["in_proj_z"]:
                D_plus_act[l_m] = [torch.zeros(4096, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(4096, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(4096, device=torch.device('cuda'))
            elif m in ["in_proj_b", "in_proj_a"]:
                D_plus_act[l_m] = [torch.zeros(32, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(32, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(32, device=torch.device('cuda'))
            elif m in ["in_proj_qkv"]:
                D_plus_act[l_m] = [torch.zeros(8192, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(8192, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(8192, device=torch.device('cuda'))
            elif m in ["gate_proj", "up_proj"]:
                D_plus_act[l_m] = [torch.zeros(9216, device=torch.device('cuda'))]
                D_minus_act[l_m] = torch.zeros(9216, device=torch.device('cuda'))
                total_sum[l_m] = torch.zeros(9216, device=torch.device('cuda'))

total_len = 0

##### main loop #######

for j in range(100):
    print(j)
    original = total_input_all[j]

    with open("original_qwen35_4B_annotated/" + str(j) + ".txt", "r", encoding="utf-8") as f:
        annotated = f.read()

    activations = {}
    inputs = tokenizer(original, return_tensors="pt")
    input_ids = inputs["input_ids"]
    _ = model(input_ids, use_cache=False)

    total_len += activations['layer_0_gate_proj'].size(0)
    for layer in range(32):
        if layer in self_attn_layer:
            temp_modules = all_modules1
        else:
            temp_modules = all_modules2
        for m in temp_modules:
            l_m = "layer_" + str(layer) + "_" + m
            D_minus_act[l_m] += torch.mean(activations[l_m], 0).cuda()

            total_sum[l_m] += activations[l_m].sum(dim=0).cuda()



    labels = ["hallucination"]
    seg_re = re.compile(r'\["(' + "|".join(labels) + r')"\]\s*(.*?)\s*\["end-section"\]', re.DOTALL)
    segments = [{"label": m.group(1), "text": m.group(2).strip()} for m in seg_re.finditer(annotated)]

    tokenizer2 = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', use_fast=True)
    enc = tokenizer2(
        original,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    offsets = enc.offset_mapping
    if len(enc['input_ids']) != activations['layer_0_gate_proj'].size()[0]:
        print("sequence dimension mismatch!!!")
        break

    threshold = 0.99

    # -----------------------------------------------------------------------------
    # hallucination loop
    # -----------------------------------------------------------------------------
    for c in range(1):
        print(hallucination[c])
        flag, target_indices = False, []
        for seg in segments:
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

            # map to token indices
            start_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s <= start_char < e
            )
            end_tok = next(
                i for i, (s, e) in enumerate(offsets) 
                if s < end_char <= e
            )

            if seg['label'] == hallucination[c]:
                flag = True
                target_indices.append([start_tok, end_tok])
                # print(original[start_char: end_char])
        if flag:
            D_plus_count[c] += 1
            rows = []
            for indices in target_indices:
                rows += list(range(indices[0]-5, indices[1]))
            for layer in range(32):
                if layer in self_attn_layer:
                    temp_modules = all_modules1
                else:
                    temp_modules = all_modules2
                for m in temp_modules:
                    l_m = "layer_" + str(layer) + "_" + m
                    D_plus_act[l_m][c] += torch.mean(activations[l_m][rows, :], 0).cuda()


    del activations

names = ["hallucination"]
for c in range(1):
    u = {}
    for layer in range(32):
        if layer in self_attn_layer:
            temp_modules = all_modules1
        else:
            temp_modules = all_modules2
        for m in temp_modules:
            l_m = "layer_" + str(layer) + "_" + m
            temp_u = D_plus_act[l_m][c] / D_plus_count[c] - D_minus_act[l_m] / len(total_input_all)
            a_overall_mean = total_sum[l_m] / total_len
            u[l_m] = temp_u * (torch.norm(a_overall_mean, p=2) / torch.norm(temp_u, p=2))
            u[l_m] = u[l_m].cpu()


    with open('all_u_ours/' + names[c] + '_u_qwen35_4B.pkl', 'wb') as f:
        pickle.dump(u, f)


