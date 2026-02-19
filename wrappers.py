import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, OPTForCausalLM, GPT2Tokenizer, pipeline
import torch
import os
import torch.nn as nn
import urllib.request
from collections import OrderedDict, defaultdict
import sentencepiece
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
    
def get_model_and_data(model_name):
    if model_name == "llama1b":
        model_id = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True)
        return model, tokenizer
        
    elif model_name == "llama3b":
        model_id = "meta-llama/Llama-3.2-3B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True)
        return model, tokenizer
        
    elif model_name == "llama7b":
        model_id = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True)
        return model, tokenizer
        
    elif model_name == "opt1.3b":
        model_id = "facebook/opt-1.3b"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True
        )
        return model, tokenizer
        
    elif model_name == "opt2.7b":
        model_id = "facebook/opt-2.7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True
        )
        return model, tokenizer
        
    elif model_name == "opt6.7b":
        model_id = "facebook/opt-6.7b"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=False
        )
        return model, tokenizer

    elif model_name == "mistral7b":
        model_id = "mistralai/Mistral-7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True)
        return model, tokenizer

    elif model_name == "vicuna7b":
        model_id = "lmsys/vicuna-7b-v1.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_safetensors=True)
        return model, tokenizer
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def get_test_data(dataset_name, tokenizer, seq_len=128, batch_size=16):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    texts = dataset["test"]["text"]
    texts = [t for t in texts if t.strip()]

    # tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=seq_len,
        return_tensors="pt"
    )

    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def save_compressed_model_only(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    if not isinstance(model, PreTrainedModel):
        print("Model is not a subclass of PreTrainedModel. Cannot save.")
        return

    model.save_pretrained(
        save_path,
        safe_serialization=True
    )

    if tokenizer:
        tokenizer.save_pretrained(save_path)

    print(f"Compressed model saved to {save_path}")
