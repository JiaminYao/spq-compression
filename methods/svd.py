import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def svd_layers_name_factory(layer_type="attention"):
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
        
def compute_cumsum_rank_ratios(model, target_layers, threshold=0.9, min_layer_size=1000):
    ratios = {}

    for name, module in model.named_modules():
        if name not in target_layers:
            continue
        if not isinstance(module, nn.Linear):
            continue

        W = module.weight.data
        if W.numel() < min_layer_size:
            print(f"Skipping {name}: too small")
            continue
        try:
            W_flat = W.view(W.size(0), -1)
            U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
            variance = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
            rank = torch.searchsorted(variance, threshold).item() + 1
            ratio = rank / min(W_flat.shape)
            ratios[name] = max(0.1, min(1-ratio, 1.0))
        except Exception as e:
            print(f"Skipping {name} due to SVD error: {e}")            
    return ratios
    
def svd_compress_layer(layer: nn.Module, rank_ratio=0.5) -> nn.Module:
    device = layer.weight.device

    if isinstance(layer, nn.Linear):
        weight = layer.weight.data
        if weight.numel() < 5000:
            return layer
        bias = layer.bias.data if layer.bias is not None else None

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        r = max(1, int(rank_ratio * min(weight.shape)))
        if r / min(weight.shape) > 0.8:
            return layer
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]

        first = nn.Linear(Vh_r.shape[1], Vh_r.shape[0], bias=False).to(device)
        second = nn.Linear(Vh_r.shape[0], U_r.shape[0], bias=(bias is not None)).to(device)

        first.weight.data = Vh_r
        second.weight.data = (U_r[:, :r] * S_r[:r])
        if bias is not None:
            second.bias.data = bias

        orig_params = weight.numel() + (bias.numel() if bias is not None else 0)
        comp_params = first.weight.numel() + second.weight.numel() + (second.bias.numel() if bias is not None else 0)
        return nn.Sequential(first, second) if comp_params < orig_params else layer

    return layer

def apply_svd_compression(model: nn.Module, rank_ratios, model_type: str, parent_name="") -> nn.Module:
    target_names = [name for name, _ in model.named_modules() if name in rank_ratios]

    for name in tqdm(target_names, desc="Applying SVD Compression", ncols=100):
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]

        last = parts[-1]
        module = getattr(parent, last)
        ratio = rank_ratios[name]

        if isinstance(module, nn.Linear):
            compressed = svd_compress_layer(module, rank_ratio=1.0 - ratio)
            setattr(parent, last, compressed)

    return model
