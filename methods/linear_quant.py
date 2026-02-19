import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class QuantLinearTensorWise(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bitwidth = bitwidth

        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("quantized_weight", torch.zeros(out_features, in_features, dtype=torch.int8))

    def quantize_from_fp(self):
        fp_max = max(self.weight_fp32.abs().max().item(), 5e-7)
        
        qmax = (1 << (self.bitwidth - 1)) - 1
        
        scale = fp_max / qmax

        self.scale = torch.tensor(scale, device=self.weight_fp32.device)
        
        self.quantized_weight = torch.round(self.weight_fp32 / self.scale).clamp(-qmax, qmax).to(torch.int8)

    def forward(self, x):
        scale = self.scale.to(x.device)
        quant_w = self.quantized_weight.to(x.device).float()

        dequant_w = quant_w * scale

        bias = self.bias.to(x.device) if self.bias is not None else None
        
        return nn.functional.linear(x, dequant_w, bias)

class QuantLinearChannelWise(nn.Module):
    def __init__(self, in_features, out_features, bias=True, bitwidth=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bitwidth = bitwidth

        self.weight_fp32 = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("scale", torch.ones(out_features))
        self.register_buffer("quantized_weight", torch.zeros(out_features, in_features, dtype=torch.int8))

    def quantize_from_fp(self):
        qmax = (1 << (self.bitwidth - 1)) - 1
        with torch.no_grad():
            for i in range(self.out_features):
                w = self.weight_fp32[i]
                max_val = max(w.abs().max().item(), 1e-6)
                self.scale[i] = max_val / qmax
                self.quantized_weight[i] = torch.round(w / self.scale[i]).clamp(-qmax, qmax).to(torch.int8)

    def forward(self, x):
        scale = self.scale.to(x.device).unsqueeze(1)
        weight = self.quantized_weight.float().to(x.device)
        dequant_weight = weight * scale
        bias = self.bias.to(x.device) if self.bias is not None else None
        return nn.functional.linear(x, dequant_weight, bias)
        
def _get_parent_and_attr(root: nn.Module, dotted: str):
    parent = root
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    return parent, parts[-1]
    
def replace_with_real_int8_per_tensor(model, bitwidth=8):
    linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    for name, old_layer in tqdm(linear_layers, desc="Applying Quantizing", ncols=100, leave=True):
        parent, attr = _get_parent_and_attr(model, name)
        qlayer = QuantLinearTensorWise(
            old_layer.in_features, old_layer.out_features, old_layer.bias is not None, bitwidth
        )
        with torch.no_grad():
            qlayer.weight_fp32.data.copy_(old_layer.weight.data)
            if old_layer.bias is not None:
                qlayer.bias.data.copy_(old_layer.bias.data)
            qlayer.quantize_from_fp()
        setattr(parent, attr, qlayer)

    return model

def replace_with_real_int8_per_channel(model, bitwidth=8):
    linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    for name, old_layer in tqdm(linear_layers, desc="Applying Quantizing", ncols=100, leave=True):
        parent, attr = _get_parent_and_attr(model, name)
        qlayer = QuantLinearChannelWise(
            old_layer.in_features, old_layer.out_features, old_layer.bias is not None, bitwidth
        )
        with torch.no_grad():
            qlayer.weight_fp32.data.copy_(old_layer.weight.data)
            if old_layer.bias is not None:
                qlayer.bias.data.copy_(old_layer.bias.data)
            qlayer.quantize_from_fp()
        setattr(parent, attr, qlayer)

    return model
    
def choose_quantization_mode_name(model, dim=0):
    per_channel = set()
    per_tensor = set()

    model_type = "unknown"
    for name, _ in model.named_modules():
        if "mlp." in name:
            model_type = "llama"
            break
        if "fc1" in name or "fc2" in name:
            model_type = "opt"
            break

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if model_type == "llama":
                if any(x in name for x in ["mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"]):
                    per_channel.add(name)
                elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    per_tensor.add(name)
                elif "embed_tokens" in name or "lm_head" in name:
                    per_tensor.add(name)
                else:
                    if module.weight.numel() > 1_000_000 or module.weight.shape[dim] >= 1024:
                        per_channel.add(name)
                    else:
                        per_tensor.add(name)

            elif model_type == "opt":
                if any(x in name for x in ["fc1", "fc2"]):
                    per_channel.add(name)
                elif any(x in name for x in ["q_proj", "k_proj", "v_proj", "out_proj"]):
                    per_tensor.add(name)
                elif "embed_tokens" in name or "lm_head" in name:
                    per_tensor.add(name)
                else:
                    if module.weight.numel() > 1_000_000 or module.weight.shape[dim] >= 1024:
                        per_channel.add(name)
                    else:
                        per_tensor.add(name)

            else:
                if module.weight.numel() > 1_000_000 or module.weight.shape[dim] >= 1024:
                    per_channel.add(name)
                else:
                    per_tensor.add(name)

    return per_channel, per_tensor

def choose_quantization_mode_percentile(model, dim=0, percentile=50):
    per_channel = set()
    per_tensor = set()

    sizes = []
    linear_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            size = module.weight.shape[dim]
            sizes.append(size)
            linear_modules[name] = module

    if not sizes:
        return per_channel, per_tensor

    cutoff = np.percentile(sizes, percentile)

    for name, module in linear_modules.items():
        if module.weight.shape[dim] >= cutoff:
            per_channel.add(name)
        else:
            per_tensor.add(name)

    return per_channel, per_tensor

def choose_quantization_mode_meanstd(model, dim=0, std_factor=0.5):
    per_channel = set()
    per_tensor = set()

    sizes = []
    linear_modules = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            size = module.weight.shape[dim]
            sizes.append(size)
            linear_modules[name] = module

    if not sizes:
        return per_channel, per_tensor

    sizes = np.array(sizes)
    mean = sizes.mean()
    std = sizes.std()
    cutoff = mean + std_factor * std

    for name, module in linear_modules.items():
        if module.weight.shape[dim] >= cutoff:
            per_channel.add(name)
        else:
            per_tensor.add(name)

    return per_channel, per_tensor

def replace_with_hybrid_quant_linear(model, per_channel_layers, per_tensor_layers, bitwidth=8):
    linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]

    with tqdm(total=len(linear_layers), desc="Applying Quantizing", ncols=100, leave=True) as pbar:
        for name, old_layer in linear_layers:
            status = "Skip"
            if name in per_channel_layers:
                parent, attr = _get_parent_and_attr(model, name)
                qlayer = QuantLinearChannelWise(
                    old_layer.in_features, old_layer.out_features, old_layer.bias is not None, bitwidth
                )
                with torch.no_grad():
                    qlayer.weight_fp32.data.copy_(old_layer.weight.data)
                    if old_layer.bias is not None:
                        qlayer.bias.data.copy_(old_layer.bias.data)
                    qlayer.quantize_from_fp()
                setattr(parent, attr, qlayer)
                status = "PC"
            elif name in per_tensor_layers:
                parent, attr = _get_parent_and_attr(model, name)
                qlayer = QuantLinearTensorWise(
                    old_layer.in_features, old_layer.out_features, old_layer.bias is not None, bitwidth
                )
                with torch.no_grad():
                    qlayer.weight_fp32.data.copy_(old_layer.weight.data)
                    if old_layer.bias is not None:
                        qlayer.bias.data.copy_(old_layer.bias.data)
                    qlayer.quantize_from_fp()
                setattr(parent, attr, qlayer)
                status = "PT"

            pbar.set_postfix_str(status)
            pbar.update(1)

    return model
