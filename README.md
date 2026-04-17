# SPQ: An Ensemble Technique for LLM Compression

<p align="center">
  <a href="https://arxiv.org/abs/2602.18420">
    <img src="https://img.shields.io/badge/arXiv-2602.18420-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C" alt="PyTorch">
  </a>
  <a href="https://huggingface.co/docs/transformers/v4.56.1/en/index">
    <img src="https://img.shields.io/badge/Transformers-4.56.1-2ECC71" alt="Transformers">
  </a>
  <a href="https://developer.nvidia.com/cuda-toolkit">
    <img src="https://img.shields.io/badge/CUDA-12.4-1ABC9C" alt="CUDA">
  </a>
  <a href="https://www.nvidia.com/en-us/data-center/a100/">
    <img src="https://img.shields.io/badge/Tested_on-2×A100_40GB-17A2B8" alt="GPU">
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/meta-llama/Llama-2-7b-hf">
    <img src="https://img.shields.io/badge/LLMs-LLaMA--2--7B-F8C471?labelColor=555555" alt="LLaMA-2-7B" />
  </a>
  <a href="https://huggingface.co/facebook/opt-6.7b">
    <img src="https://img.shields.io/badge/LLMs-OPT--6.7B-85C1E9?labelColor=555555" alt="OPT-6.7B" />
  </a>
  <a href="https://huggingface.co/mistralai/Mistral-7B-v0.1">
    <img src="https://img.shields.io/badge/LLMs-Mistral--7B-D7BDE2?labelColor=555555" alt="Mistral-7B" />
  </a>
  <a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">
    <img src="https://img.shields.io/badge/LLMs-Vicuna--7B-A3E4D7?labelColor=555555" alt="Vicuna-7B" />
  </a>
</p>

<p align="center">
  <b>Accepted to LREC 2026 Main Conference</b>
  <br>
  <a href="https://arxiv.org/abs/2602.18420">Paper</a>
</p>

## Introduction

SPQ is an ensemble compression framework that combines three complementary techniques to reduce the size of large language models while preserving generation quality:

- **Variance-Retained Low-Rank SVD** — decomposes weight matrices into low-rank approximations, retaining a configurable amount of variance.
- **Activation-based Structured Pruning** — removes less important attention heads and MLP neurons based on activation magnitudes.
- **INT8 Linear Quantization** — quantizes remaining weights to 8-bit integers with per-channel or mixed-granularity scaling.

## Quick Start

### 1. Environment Setup

```bash
conda create -n spq python=3.11
conda activate spq
git clone https://github.com/JiaminYao/spq-compression.git
cd spq-compression
pip install -r requirements.txt
```

One GPU with at least 40GB VRAM is required (e.g., A100 40GB).

### 2. Hugging Face Access

Some models require gated access (e.g., LLaMA, Mistral). Create a token with at least read permission at https://huggingface.co/settings/tokens, then log in:

```bash
huggingface-cli login
```

### 3. Run

```bash
python main.py
```

Custom example with recommended settings:

```bash
python main.py \
  --model_name llama7b \
  --max_prune_ratio 0.05 \
  --variance_threshold 0.96 \
  --quant_mode mix_percentile \
  --steps 200
```

## Supported Models

Use these strings for the `--model_name` argument:

| model_name  | Hugging Face ID |
|-------------|------------------|
| llama1b     | meta-llama/Llama-3.2-1B |
| llama3b     | meta-llama/Llama-3.2-3B |
| llama7b     | meta-llama/Llama-2-7b-hf |
| opt1.3b     | facebook/opt-1.3b |
| opt2.7b     | facebook/opt-2.7b |
| opt6.7b     | facebook/opt-6.7b |
| mistral7b   | mistralai/Mistral-7B-v0.1 |
| vicuna7b    | lmsys/vicuna-7b-v1.5 |

## Arguments

| Argument | Description |
|----------|-------------|
| `--model_name` | Pretrained model to load (default: `llama7b`) |
| `--max_prune_ratio` | Maximum pruning ratio, 0–1 |
| `--variance_threshold` | SVD retained variance, 0–1 |
| `--quant_mode` | `channel` / `tensor` / `mix_name` / `mix_percentile` / `mix_adaptive` |
| `--steps` | LoRA fine-tuning steps |

## Pipeline Output

| Metric | Description |
|--------|-------------|
| Weight Memory | Model size before vs. after compression |
| Perplexity | Language modeling quality before vs. after |
| Throughput | Tokens/sec inference speed before vs. after |

## Citation

If you find this work useful, please cite:

```bibtex
@article{yao2026spq,
  title={SPQ: An Ensemble Technique for Large Language Model Compression},
  author={Yao, Jiamin and Gultepe, Eren},
  journal={arXiv preprint arXiv:2602.18420},
  year={2026}
}
```
