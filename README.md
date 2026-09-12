# Vision Transformer on CIFAR-10

A PyTorch implementation of a Vision Transformer (ViT-Tiny)
adapted for image classification on CIFAR-10.

## Overview

This project implements the core components of a Vision Transformer
from scratch, including patch embedding, multi-head self-attention,
transformer encoder blocks, positional embeddings, and classification.

The implementation is adapted for the small 32×32 CIFAR-10 images
using 4×4 image patches.

## Architecture

| Component | Configuration |
|---|---|
| Image size | 32×32 |
| Patch size | 4×4 |
| Number of patches | 64 |
| Embedding dimension | 192 |
| Transformer depth | 6 |
| Attention heads | 6 |
| MLP ratio | 4 |
| Number of classes | 10 |

## Implementation

The model consists of:

- Convolutional patch embedding
- Learnable [CLS] token
- Learnable positional embeddings
- Multi-head self-attention
- Pre-LayerNorm transformer blocks
- GELU MLP
- Linear classification head

## Dataset

CIFAR-10 is used for training and evaluation.

Training uses random cropping and horizontal flipping,
while the test set uses normalization without augmentation.

## Training

- Optimizer: AdamW
- Learning rate: 5e-4
- Weight decay: 1e-2
- Label smoothing: 0.1
- Epochs: 40
- Batch size: 128
- Learning-rate schedule: linear warmup + cosine decay
- Gradient clipping: 1.0

## Project Structure

```text
.
├── dataset.py
├── model.py
├── train.py
├── README.md
└── .gitignore

## Running


pip install -r requirements.txt
python train.py
```


## Reference

This implementation was developed by studying and taking inspiration from the Vision Transformer implementation by Shivam Raj Sharma.

The reference implementation was used to understand the overall ViT architecture and training approach, while this repository implements and adapts the model for CIFAR-10.


## Status

The ViT architecture and training pipeline are implemented.

Extensive benchmarking across different training configurations is outside the scope of this repository.
