import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def prune_layers_name_factory(layer_type="mlp"):
    def attention_layers(model_type: str, num_layers: int):
        layers = []
        if model_type == "llama":
            for i in range(num_layers):
                layers.extend([
                    f"model.layers.{i}.self_attn.q_proj",
                    f"model.layers.{i}.self_attn.k_proj",
                    f"model.layers.{i}.self_attn.v_proj",
                    f"model.layers.{i}.self_attn.o_proj",
                ])
        elif model_type == "opt":
            for i in range(num_layers):
                layers.extend([
                    f"model.decoder.layers.{i}.self_attn.q_proj",
                    f"model.decoder.layers.{i}.self_attn.k_proj",
                    f"model.decoder.layers.{i}.self_attn.v_proj",
                    f"model.decoder.layers.{i}.self_attn.out_proj"
                ])
        return layers

    def mlp_layers(model_type: str, num_layers: int):
        layers = []
        if model_type == "llama":
            for i in range(num_layers):
                layers.extend([
                    f"model.layers.{i}.mlp.gate_proj",
                    f"model.layers.{i}.mlp.down_proj",
                    f"model.layers.{i}.mlp.up_proj",
                ])
        elif model_type == "opt":
            for i in range(num_layers):
                layers.extend([
                    f"model.decoder.layers.{i}.fc1",
                    f"model.decoder.layers.{i}.fc2",
                ])
        return layers

    def all_layers(model_type: str, num_layers: int):
        return attention_layers(model_type, num_layers) + mlp_layers(model_type, num_layers)

    if layer_type == "attention":
        return attention_layers
    elif layer_type == "mlp":
        return mlp_layers
    elif layer_type == "all":
        return all_layers
    else:
        raise ValueError(f"Unknown layer_type: {layer_type}")

def get_num_heads_and_head_dim(attn_module):
    if hasattr(attn_module, "num_heads") and hasattr(attn_module, "head_dim"):
        return attn_module.num_heads, attn_module.head_dim

    if hasattr(attn_module, "n_head") and hasattr(attn_module, "head_dim"):
        return attn_module.n_head, attn_module.head_dim

    cfg = getattr(attn_module, "config", None)
    if cfg is not None:
        nh = getattr(cfg, "num_attention_heads", None)
        hs = getattr(cfg, "hidden_size", None)
        if nh and hs:
            return nh, hs // nh

    if hasattr(attn_module, "q_proj"):
        weight = attn_module.q_proj.weight
        if len(weight.shape) == 2:
            total_dim = weight.shape[0]
            default_heads = 32
            return default_heads, total_dim // default_heads

    raise AttributeError("Cannot find num_heads or head_dim in attn_module or infer from weights")

def collect_activation_stats(model: nn.Module, target_layer_names, dataloader, batch_size, mode="l2", device="cuda"):
    model.eval()
    activation_means = {name: [] for name in target_layer_names}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if output.dim() >= 2:
                if mode == "l1":
                    act = output.abs().mean(dim=tuple(range(output.dim() - 1)))
                elif mode == "l2":
                    act = output.pow(2).mean(dim=tuple(range(output.dim() - 1)))
                else:
                    raise ValueError(f"Unsupported mode: {mode}. Use 'l1' or 'l2'.")
            else:
                if mode == "l1":
                    act = output.abs().mean()
                elif mode == "l2":
                    act = output.pow(2).mean()
            activation_means[name].append(act.detach().cpu())
        return hook_fn

    for name, module in model.named_modules():
        if name in target_layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
            else:
                input_ids = batch[0].to(device)
            model(input_ids)
            if i + 1 >= batch_size:
                break

    for h in hooks:
        h.remove()

    for name in activation_means:
        acts = activation_means[name]
        if len(acts) == 0:
            print(f"Warning: No activations recorded for layer {name}")
            activation_means[name] = None
        else:
            activation_means[name] = torch.stack(acts).mean(dim=0)

    return activation_means

def get_per_layer_prune_ratios(activation_means: dict, layers: int, prune_mode: str, model_type: str = "llama", max_ratio=0.10, min_ratio=0.00):
    act_means = []
    
    for i in range(layers):
        if model_type == "llama":
            if prune_mode in ("mlp", "all"):
                key = f"model.layers.{i}.mlp.up_proj"
            elif prune_mode == "attention":
                key = f"model.layers.{i}.self_attn.k_proj"
        elif model_type == "opt":
            if prune_mode in ("mlp", "all"):
                key = f"model.decoder.layers.{i}.fc1"
            elif prune_mode == "attention":
                key = f"model.decoder.layers.{i}.self_attn.k_proj"
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        act_val = activation_means.get(key, None)
        if act_val is None:
            act_mean = 0.0
        else:
            act_mean = act_val.mean().item()
        act_means.append(act_mean)

    act_means = np.array(act_means)
    log_vals = np.log(act_means + 1e-6)
    log_norm = (log_vals.max() - log_vals) / (log_vals.max() - log_vals.min() + 1e-8)
    layer_ratios = min_ratio + log_norm * (max_ratio - min_ratio)

    return layer_ratios

def prune_mlp_by_activation(linear_layer: nn.Linear, activation_scores: torch.Tensor, prune_ratio: float, model_type: str, device="cuda"):
    total_neurons = activation_scores.numel()
    keep_neurons = int(total_neurons * (1 - prune_ratio))
    _, topk_idx = torch.topk(activation_scores, keep_neurons)
    topk_idx, _ = torch.sort(topk_idx)

    in_features = linear_layer.in_features
    out_features = linear_layer.out_features
    has_bias = linear_layer.bias is not None

    if activation_scores.shape[0] == out_features:
        new_linear = nn.Linear(in_features, keep_neurons, bias=has_bias)
        with torch.no_grad():
            new_linear.weight.copy_(linear_layer.weight[topk_idx, :])
            if has_bias:
                new_linear.bias.copy_(linear_layer.bias[topk_idx])

    elif activation_scores.shape[0] == in_features:
        new_linear = nn.Linear(keep_neurons, out_features, bias=has_bias)
        with torch.no_grad():
            new_linear.weight.copy_(linear_layer.weight[:, topk_idx])
            if has_bias:
                new_linear.bias.copy_(linear_layer.bias)

    else:
        raise ValueError("Activation score dimension does not match layer weight shape.")

    new_linear = new_linear.to(device)
    linear_layer.__class__ = new_linear.__class__
    linear_layer.__dict__.update(new_linear.__dict__)

def prune_attention_heads(attn_module, head_importance_scores, prune_ratio, device="cuda"):
    num_heads, head_dim = get_num_heads_and_head_dim(attn_module)

    total_heads_to_keep = int(num_heads * (1 - prune_ratio))
    if total_heads_to_keep == 0:
        raise ValueError("Prune ratio too high, no heads left to keep!")

    topk_heads = torch.topk(head_importance_scores, total_heads_to_keep).indices.sort().values

    keep_indices = []
    for head_idx in topk_heads:
        keep_indices.extend(range(head_idx * head_dim, (head_idx + 1) * head_dim))
    keep_indices = torch.tensor(keep_indices, device=attn_module.q_proj.weight.device)

    def prune_proj(proj_layer, direction="row"):
        in_features = proj_layer.in_features
        out_features = proj_layer.out_features
        has_bias = proj_layer.bias is not None

        if direction == "row":
            new_layer = nn.Linear(in_features, len(keep_indices), bias=has_bias)
            with torch.no_grad():
                new_layer.weight.copy_(proj_layer.weight[keep_indices, :])
                if has_bias:
                    new_layer.bias.copy_(proj_layer.bias[keep_indices])
        else:
            new_layer = nn.Linear(len(keep_indices), out_features, bias=has_bias)
            with torch.no_grad():
                new_layer.weight.copy_(proj_layer.weight[:, keep_indices])
                if has_bias:
                    new_layer.bias.copy_(proj_layer.bias)
        return new_layer.to(device)

    if hasattr(attn_module, "o_proj"):
        attn_module.q_proj = prune_proj(attn_module.q_proj, "row")
        attn_module.k_proj = prune_proj(attn_module.k_proj, "row")
        attn_module.v_proj = prune_proj(attn_module.v_proj, "row")
        attn_module.o_proj = prune_proj(attn_module.o_proj, "col")
    
    elif hasattr(attn_module, "out_proj"):
        attn_module.q_proj = prune_proj(attn_module.q_proj, "row")
        attn_module.k_proj = prune_proj(attn_module.k_proj, "row")
        attn_module.v_proj = prune_proj(attn_module.v_proj, "row")
        attn_module.out_proj = prune_proj(attn_module.out_proj, "col")

    attn_module.num_heads = total_heads_to_keep
    attn_module.head_dim = head_dim
    attn_module.hidden_size = total_heads_to_keep * head_dim

def prune_model_by_activation(model: nn.Module, activation_means: dict, layer_ratios, prune_mode: str, model_type: str, num_layers: int, device="cuda"):
    named_modules = dict(model.named_modules())

    if isinstance(layer_ratios, (float, int)):
        layer_ratios = [layer_ratios] * num_layers

    desc = {
        "mlp": "Applying Pruning",
        "attention": "Applying Pruning",
        "all": "Applying Pruning",
    }.get(prune_mode, f"Pruning ({prune_mode})")

    if prune_mode == "mlp":
        for i in tqdm(range(num_layers), desc=desc, ncols=100, leave=True):
            prune_ratio = layer_ratios[i]

            if model_type == "llama":
                mlp = named_modules.get(f"model.layers.{i}.mlp", None)
                if mlp is None:
                    continue
                act_scores = activation_means.get(f"model.layers.{i}.mlp.up_proj", None)
                if act_scores is None:
                    continue

                prune_mlp_by_activation(mlp.up_proj, act_scores, prune_ratio, model_type, device)
                if hasattr(mlp, "gate_proj"):
                    prune_mlp_by_activation(mlp.gate_proj, act_scores, prune_ratio, model_type, device)
                prune_mlp_by_activation(mlp.down_proj, act_scores, prune_ratio, model_type, device)

            elif model_type == "opt":
                fc1_name = f"model.decoder.layers.{i}.fc1"
                fc2_name = f"model.decoder.layers.{i}.fc2"
                if fc1_name not in named_modules or fc2_name not in named_modules:
                    continue
                act_scores = activation_means.get(fc1_name, None)
                if act_scores is None:
                    continue

                prune_mlp_by_activation(named_modules[fc1_name], act_scores, prune_ratio, model_type, device)
                prune_mlp_by_activation(named_modules[fc2_name], act_scores, prune_ratio, model_type, device)

            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

    elif prune_mode == "attention":
        for i in tqdm(range(num_layers), desc=desc, ncols=100, leave=False):
            prune_ratio = layer_ratios[i]

            if model_type == "llama":
                q_name = f"model.layers.{i}.self_attn.q_proj"
                k_name = f"model.layers.{i}.self_attn.k_proj"
                v_name = f"model.layers.{i}.self_attn.v_proj"
                o_name = f"model.layers.{i}.self_attn.o_proj"

                if not all(n in activation_means and activation_means[n] is not None for n in [q_name, k_name, v_name]):
                    continue
                if not all(n in named_modules for n in [q_name, k_name, v_name, o_name]):
                    continue

                avg_score = (activation_means[q_name] + activation_means[k_name] + activation_means[v_name]) / 3.0
                attn_module = named_modules[f"model.layers.{i}.self_attn"]

                num_heads, head_dim = get_num_heads_and_head_dim(attn_module)
                head_scores = avg_score.view(num_heads, head_dim).mean(dim=1)

                prune_attention_heads(attn_module, head_scores, prune_ratio, device)

            elif model_type == "opt":
                q_name = f"model.decoder.layers.{i}.self_attn.q_proj"
                k_name = f"model.decoder.layers.{i}.self_attn.k_proj"
                v_name = f"model.decoder.layers.{i}.self_attn.v_proj"
                o_name = f"model.decoder.layers.{i}.self_attn.out_proj"

                if not all(n in activation_means and activation_means[n] is not None for n in [q_name, k_name, v_name]):
                    continue
                if not all(n in named_modules for n in [q_name, k_name, v_name, o_name]):
                    continue

                avg_score = (activation_means[q_name] + activation_means[k_name] + activation_means[v_name]) / 3.0
                attn_module = named_modules[f"model.decoder.layers.{i}.self_attn"]

                num_heads, head_dim = get_num_heads_and_head_dim(attn_module)
                head_scores = avg_score.view(num_heads, head_dim).mean(dim=1)

                prune_attention_heads(attn_module, head_scores, prune_ratio, device)

            else:
                raise ValueError(f"Unsupported model_type: {model_type}")

    elif prune_mode == "all":
        mlp_ratios = [layer_ratios[i] for i in range(num_layers)]
        attn_ratios = [layer_ratios[i] for i in range(num_layers)]

        prune_model_by_activation(model, activation_means, mlp_ratios, "mlp", model_type, num_layers=num_layers, device=device)
        prune_model_by_activation(model, activation_means, attn_ratios, "attention", model_type, num_layers=num_layers, device=device)

    else:
        raise ValueError(f"Unknown prune_mode: {prune_mode}")
