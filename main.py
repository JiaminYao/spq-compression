import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import gc
import torch
import torch.nn as nn
import numpy as np
import random
import math
from datasets import load_dataset
from wrappers import get_model_and_data, get_test_data
from eval_utils import *
from config import MODEL_CONFIG
from methods.train import *
from methods.pruning import *
from methods.linear_quant import *
from methods.svd import *
import argparse

def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(model_name="llama7b", max_prune_ratio=0.05, variance_threshold = 0.96, quant_mode = "mix_name", steps=200):
    model_data = get_model_and_data(model_name)
    if model_data is None:
        print(f"Error: Unable to load model data for {model_name}")
        return
    
    model, tokenizer = model_data
    cfg = MODEL_CONFIG[model_name]
    layers = cfg["num_layers"]

    device = torch.device("cuda:0")
    device2 = torch.device("cuda:1")
    
    model = model.to(device)
    # print(model.dtype)
    
    if model_name.lower().startswith("opt"):
        model_type = "opt"
    else:
        model_type = "llama"

    original_state_dict = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}

    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["test"]
    sample_text = "\n".join([x["text"] for x in test_dataset.select(range(100))])

    min_prune_ratio = 0.00
    dataloader_batch_size = 1
    acti_batch_size = 8
    activation_mode = "l1"
    prune_layers_name="mlp"
    svd_layers_name="attention"
    prune_layers_func = prune_layers_name_factory(layer_type=prune_layers_name)
    prune_layers = prune_layers_func(model_type, layers)
    prune_layer_types = {'.'.join(name.split('.')[-2:]) for name in prune_layers}
    svd_layers_fn = svd_layers_name_factory(layer_type=svd_layers_name)
    svd_layers = svd_layers_fn(model_type=model_type, num_layers=layers)
    svd_layer_types = {'.'.join(name.split('.')[-2:]) for name in svd_layers}
    
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Prune Ratio Range: {min_prune_ratio}-{max_prune_ratio}")
    print(f"Variance Threshold: {variance_threshold}")
    print(f"Linear Quantization Mode: {quant_mode}")
    print(f"Compression Settings -> Low-rank SVD ✅ | Structured Pruning ✅ | Linear Quantization ✅")
    print("=" * 60)    

    set_seed(42)
    
    dataloader = get_test_data("wikitext2", tokenizer, seq_len=128, batch_size = dataloader_batch_size)

    print(f"Evaluating original model")
    mem_before = get_mem_size_gb(model)
    ppl_before = calculate_perplexity(model, tokenizer, sample_text, device=device)
    thr_before = evaluate_throughput(model, tokenizer, sample_text, device)

    energy_rank_ratios = compute_cumsum_rank_ratios(model, svd_layers, threshold=variance_threshold, min_layer_size=1000)
    model = apply_svd_compression(model, rank_ratios=energy_rank_ratios, model_type=model_type)
    
    activation_means = collect_activation_stats(
        model=model,
        target_layer_names=prune_layers,
        dataloader=dataloader,
        batch_size=acti_batch_size,
        mode=activation_mode,
        device=device
    )
    
    layer_ratios = get_per_layer_prune_ratios(
        activation_means, 
        layers=layers, 
        prune_mode=prune_layers_name,
        model_type=model_type, 
        max_ratio=max_prune_ratio, 
        min_ratio=min_prune_ratio
    )

    prune_model_by_activation(
        model, 
        activation_means, 
        layer_ratios, 
        prune_mode=prune_layers_name, 
        model_type=model_type, 
        num_layers=layers, 
        device=device
    )
      
    if quant_mode == "tensor":
        model = replace_with_real_int8_per_tensor(model, bitwidth=8)
    elif quant_mode == "mix_name":
        per_channel_layers, per_tensor_layers = choose_quantization_mode_name(model)
        model = replace_with_hybrid_quant_linear(model, per_channel_layers, per_tensor_layers, bitwidth=8)
    elif quant_mode == "mix_percentile":
        per_channel_layers, per_tensor_layers = choose_quantization_mode_percentile(model, percentile=50)
        model = replace_with_hybrid_quant_linear(model, per_channel_layers, per_tensor_layers, bitwidth=8)
    elif quant_mode == "mix_meanstd":
        per_channel_layers, per_tensor_layers = choose_quantization_mode_meanstd(model, dim=0, std_factor=0.5)
        model = replace_with_hybrid_quant_linear(model, per_channel_layers, per_tensor_layers, bitwidth=8)
    elif quant_mode == "channel":
        model = replace_with_real_int8_per_channel(model, bitwidth=8) 
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")

    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device2)

    inject_lora_adapters(model, model_type=model_type)
    fine_tune(model, tokenizer, device=device2, max_steps=steps)

    print(f"Evaluating after SPQ compression")
    mem_after = get_mem_size_gb(model)
    ppl_after = calculate_perplexity(model.eval(), tokenizer, sample_text, device=device2)
    thr_after = evaluate_throughput(model.eval(), tokenizer, sample_text, device=device2)
    
    print(f"[Before Compression] MEM Size: {mem_before:,.2f} GB")
    print(f"[After Compression]  MEM Size: {mem_after:,.2f} GB")
    print(f"Compression Ratio: {(1-(mem_after / mem_before)) * 100:.2f}%")

    print(f"[Before Compression] Perplexity: {ppl_before:,.2f}")
    print(f"[After Compression]  Perplexity: {ppl_after:,.2f}")
    print(f"Perplexity Change: {ppl_after - ppl_before:,.2f}")
    
    print(f"[Before Compression] Throughput: {thr_before:.2f} tokens/sec")
    print(f"[After Compression]  Throughput: {thr_after:.2f} tokens/sec")
    print(f"Throughput Change: {thr_after - thr_before:.2f} tokens/sec")

    try:
        del model
        del tokenizer
    except NameError:
        pass
    clear_memory()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPQ pipeline")

    parser.add_argument(
        "--model_name",
        type=str,
        default="llama7b",
        help="Model name, e.g. llama7b / opt1.3b / opt2.7b",
    )
    parser.add_argument(
        "--max_prune_ratio",
        type=float,
        default=0.05,
        help="Maximum structured pruning ratio per layer",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=0.96,
        help="SVD energy (variance) retention threshold (0–1)",
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        default="mix_name",
        choices=["tensor", "mix_name", "mix_percentile", "mix_meanstd", "channel"],
        help="Linear quantization mode",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of LoRA fine-tuning steps",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        max_prune_ratio=args.max_prune_ratio,
        variance_threshold=args.variance_threshold,
        quant_mode=args.quant_mode,
        steps=args.steps,
    )
