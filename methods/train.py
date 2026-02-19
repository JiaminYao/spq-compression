import torch
import torch.nn as nn
from tqdm import tqdm
import math
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch.nn.functional as F
from itertools import islice
import random
import warnings
warnings.filterwarnings("ignore", message="`torch.cuda.amp.GradScaler", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast", category=FutureWarning)

def fine_tune(model, tokenizer, device="cuda", max_steps=200):
    model = model.to(device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable()

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not trainable_params:
        raise ValueError("No trainable parameters found in the model.")

    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]
    dataset = dataset.shuffle(seed=42).select(range(10000))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    steps_done = 0
    with tqdm(total=max_steps, desc="Fine-tuning", ncols=100) as pbar:
        for batch in dataloader:
            if steps_done >= max_steps:
                break

            text = batch["text"][0]
            if not text.strip():
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            padding_mask = inputs["attention_mask"] == 0
            labels[padding_mask] = -100

            with torch.cuda.amp.autocast():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps_done += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if steps_done >= max_steps:
                break
                
    model.eval()
    return model

class LoRALinearWrapper(nn.Module):
    def __init__(self, base_layer, r=8, alpha=32):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        in_features, out_features = self._get_layer_dimensions(base_layer)

        # LORA para
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / r

    def forward(self, x):
        lora_delta = F.linear(x, self.lora_B @ self.lora_A) * self.scaling
        return self.base_layer(x) + lora_delta

    def _get_layer_dimensions(self, layer):
        if isinstance(layer, nn.Linear):
            return layer.in_features, layer.out_features

        if isinstance(layer, nn.Sequential):
            first = layer[0]
            second = layer[1]
            if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                return first.in_features, second.out_features
            else:
                raise ValueError(f"Unsupported Sequential sub-layers: {type(first)}, {type(second)}")

        if "QuantLinearTensorWise" in str(type(layer)) or "QuantLinearChannelWise" in str(type(layer)):
            return layer.in_features, layer.out_features

        raise ValueError(f"Unsupported layer type: {type(layer)}")

def inject_lora_adapters(model: nn.Module, model_type="opt", r=8, alpha=32, force=False):
    if model_type == "llama":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    elif model_type == "opt":
        target_modules = ["fc1", "fc2"]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    injected_count = 0
    skipped = []

    def get_parent_module(model, name):
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            try:
                idx = int(part)
                parent = parent[idx]
            except ValueError:
                parent = getattr(parent, part)
        return parent, parts[-1]

    for name, module in model.named_modules():
        if any(substr in name for substr in target_modules):
            try:
                parent, attr_name = get_parent_module(model, name)
                old_layer = getattr(parent, attr_name)

                if isinstance(old_layer, LoRALinearWrapper):
                    if force:
                        base_layer = old_layer.base_layer
                    else:
                        skipped.append(name + " (already LoRA)")
                        continue
                else:
                    base_layer = old_layer

                lora_layer = LoRALinearWrapper(base_layer, r=r, alpha=alpha)
                setattr(parent, attr_name, lora_layer)
                injected_count += 1
            except Exception as e:
                skipped.append(f"{name} (error: {e})")

def freeze_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "lora_" not in name and "LoRALinearWrapper" not in name:
            param.requires_grad = False
