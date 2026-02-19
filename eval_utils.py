import os
import gc
import time
import math
import csv
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
from evaluate import load as load_metric
from methods.linear_quant import QuantLinearTensorWise, QuantLinearChannelWise

def get_mem_size_mb(model):
    total_bits = 0

    for module in model.modules():
        if isinstance(module, (QuantLinearTensorWise, QuantLinearChannelWise)):
            total_bits += module.quantized_weight.numel() * module.bitwidth
            if module.bias is not None:
                total_bits += module.bias.numel() * 32
        else:
            for param in module.parameters(recurse=False):
                if param.dtype == torch.float16:
                    total_bits += param.numel() * 16
                elif param.dtype == torch.float32:
                    total_bits += param.numel() * 32
                else:
                    total_bits += param.numel() * torch.finfo(param.dtype).bits

    return total_bits / 8 / 1024 / 1024

def get_mem_size_gb(model):
    total_bits = 0

    for module in model.modules():
        if isinstance(module, (QuantLinearTensorWise, QuantLinearChannelWise)):
            total_bits += module.quantized_weight.numel() * module.bitwidth
            if module.bias is not None:
                total_bits += module.bias.numel() * 32
        else:
            for param in module.parameters(recurse=False):
                if param.dtype == torch.float16:
                    total_bits += param.numel() * 16
                elif param.dtype == torch.float32:
                    total_bits += param.numel() * 32
                else:
                    total_bits += param.numel() * torch.finfo(param.dtype).bits

    return total_bits / 8 / 1e9
    
def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def count_model_params_quan(model):
    total_bits = 0
    for module in model.modules():
        if hasattr(module, "quantized_weight"):
            total_bits += module.quantized_weight.numel() * module.bitwidth
        if hasattr(module, "bias") and module.bias is not None:
            total_bits += module.bias.numel() * 32
    return total_bits // 32

def calculate_perplexity(model, tokenizer, text, device="cuda"):

    model.eval()

    model_dtype = next(model.parameters()).dtype

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels = inputs["input_ids"].clone()

    padding_mask = inputs["attention_mask"] == 0
    labels[padding_mask] = -100

    inputs = {
        k: (v.to(device).to(model_dtype) if v.dtype.is_floating_point else v.to(device))
        for k, v in inputs.items()
    }
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

    try:
        return math.exp(loss.item())
    except OverflowError:
        return float("inf")

def evaluate_throughput(model, tokenizer, text, device="cuda", max_new_tokens=64, num_trials=10, use_sampling=False):
    model.eval()
    model_dtype = next(model.parameters()).dtype

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = {
        k: (v.to(device).to(model_dtype) if v.dtype.is_floating_point else v.to(device))
        for k, v in inputs.items()
    }

    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=use_sampling,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    total_tokens = 0
    total_time = 0.0

    with torch.no_grad():
        for _ in range(num_trials):
            torch.cuda.synchronize()
            start = time.time()

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=use_sampling,
                top_k=50,
                top_p=0.9,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )

            torch.cuda.synchronize()
            end = time.time()

            generated = outputs.shape[-1] - input_ids.shape[-1]
            total_tokens += generated
            total_time += (end - start)

    return total_tokens / total_time if total_time > 0 else 0.0
