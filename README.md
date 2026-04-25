# NanoGPT Shakespeare Language Model

A detailed implementation of a GPT-style character-level language model inspired by nanoGPT, built from scratch using PyTorch. The model is trained on Shakespeare text data to learn language structure, dialogue flow, punctuation patterns, and stylistic characteristics of classical English writing.

This project was created to understand transformer architectures at a systems level rather than only using pretrained APIs. It focuses on implementing the internal mechanics of modern language models manually: tokenization, embeddings, masked attention, transformer blocks, optimization, and autoregressive generation.

---

# Table of Contents

1. Project Overview
2. Key Features
3. Technology Stack
4. Repository Structure
5. Dataset
6. How the Model Works
7. Transformer Components
8. Training Process
9. Text Generation
10. Hyperparameters
11. Learning Outcomes
12. Challenges Solved
13. Future Improvements
14. How to Run
15. Credits
16. License

---

# Project Overview

Large Language Models such as GPT rely on transformer architectures capable of learning long-range dependencies in sequential data. This project recreates those core ideas in a smaller educational setting using character-level modeling.

Instead of predicting words or subwords, the model predicts the next character in a sequence. Despite the simplicity of character-level training, the system learns sentence structure, formatting, names, punctuation, and style from raw text alone.

The final model can generate new Shakespeare-like text autoregressively, one token at a time.

---

# Key Features

* Built a decoder-only transformer architecture from scratch
* Implemented causal masked self-attention
* Implemented multi-head attention mechanism
* Used trainable token embeddings
* Used trainable positional embeddings
* Added feed-forward neural network layers
* Added residual connections and Layer Normalization
* Trained with AdamW optimizer
* Used train/validation split for evaluation
* Implemented autoregressive text generation
* Supports GPU acceleration with CUDA when available
* Hyperparameter experimentation for performance tuning
* Clean modular PyTorch implementation

---

# Technology Stack

* Python
* PyTorch
* Jupyter Notebook / VS Code
* CUDA (optional GPU acceleration)

---

# Repository Structure

```text
.
├── README.md              # Project documentation
├── LICENSE                # Open-source license
├── input.txt              # Shakespeare dataset
├── clean_input.py         # Dataset cleaning / preprocessing utilities
├── bbg_Bigram.py          # Language model training script
```

---

# Dataset

The project uses a Shakespeare text corpus commonly used for language modeling experiments.

The dataset contains:

* Character dialogues
  n- Monologues
* Stage directions
* Names of characters
* Classical grammar and punctuation patterns
* Rich stylistic variation

This makes it ideal for sequence prediction and generative modeling tasks.

---

# How the Model Works

The objective of the model is simple:

> Given previous characters, predict the next character.

Example:

```text
Input : TO BE OR NOT T
Target: O BE OR NOT TO
```

The model repeatedly learns this next-token prediction task over many batches of training data.

---

# Data Preprocessing

## 1. Vocabulary Construction

All unique characters in the dataset are collected and sorted.

Example:

```python
chars = sorted(list(set(text)))
```

Each character is assigned an integer ID.

* `stoi` = string to index
* `itos` = index to string

---

## 2. Encoding

Raw text is converted into token IDs.

```python
encoded = [stoi[c] for c in text]
```

---

## 3. Train / Validation Split

The dataset is divided into:

* 90% training data
* 10% validation data

This helps evaluate generalization.

---

## 4. Batch Sampling

Random context windows are sampled during training.

Each batch contains:

* Input tokens
* Target tokens shifted by one position

---

# Transformer Components

# 1. Token Embeddings

Maps each character ID into a dense trainable vector space.

```text
Token ID -> Embedding Vector
```

This allows semantic learning beyond one-hot encoding.

---

# 2. Positional Embeddings

Transformers have no built-in notion of order, so positional embeddings are added to token embeddings.

This allows the model to understand sequence position.

---

# 3. Self-Attention

Self-attention allows each token to selectively focus on previous relevant tokens.

The model computes:

* Query (Q)
* Key (K)
* Value (V)

Attention scores are calculated as:

```text
softmax(QK^T / sqrt(d))
```

Where:

* `QK^T` measures similarity
* `sqrt(d)` stabilizes scale
* `softmax` converts scores into probabilities

---

# 4. Causal Masking

Future tokens must remain hidden during training.

A lower triangular mask ensures token `t` only attends to positions `<= t`.

This preserves autoregressive behavior.

---

# 5. Multi-Head Attention

Multiple attention heads learn different relationships simultaneously.

Examples:

* Syntax relationships
* Long-range references
* Local context patterns
* Formatting structure

Outputs from all heads are concatenated and projected.

---

# 6. Feed Forward Network

Each token representation is passed through a small MLP:

```text
Linear -> ReLU -> Linear
```

This improves representational power.

---

# 7. Residual Connections

Residual paths help preserve information and improve optimization.

```text
x = x + layer(x)
```

---

# 8. Layer Normalization

Used to stabilize activations and training dynamics.

---

# Training Process

The model is trained using mini-batch gradient descent.

## Loss Function

Cross-entropy loss compares predicted next-token probabilities with true targets.

## Optimizer

AdamW is used for efficient and stable optimization.

## Evaluation

Training and validation loss are periodically estimated.

Example:

```text
step 0: train loss 4.21 | val loss 4.24
step 500: train loss 2.35 | val loss 2.47
```

Lower loss indicates better predictions.

---

# Text Generation

After training, generation begins from an initial context token.

At each step:

1. Predict next token probabilities
2. Sample next token
3. Append token to context
4. Repeat

This creates new text autoregressively.

---

# Example Hyperparameters

```python
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

These settings balance training speed and quality.

---

# Learning Outcomes

Through this project, I gained hands-on understanding of:

* Transformer internals
* Attention mathematics
* Tensor dimensions and matrix multiplication
* Autoregressive generation
* Sequence modeling
* Neural network optimization
* Numerical precision issues
* GPU/CPU training workflows
* Debugging PyTorch runtime errors
* Practical experimentation methodology

---

# Challenges Solved

* PyTorch installation and environment setup
* Kernel mismatches in Jupyter
* Tensor shape debugging
* Attention matrix dimension issues
* Floating point precision confusion
* Hyperparameter tuning tradeoffs
* Training stability concerns

---

# Future Improvements

* Byte Pair Encoding (BPE) tokenization
* Larger and cleaner datasets
* Model checkpoint saving/loading
* Better decoding (temperature, top-k, nucleus sampling)
* Mixed precision training
* Learning rate schedulers
* Validation metrics dashboard
* Web UI deployment using Streamlit or Gradio
* Quantized inference for speed
* Fine-tuning on custom corpora

---

# How to Run

## 1. Install Dependencies

```bash
pip install torch
```

## 2. Add Dataset

Place `input.txt` in the root project folder.

## 3. Run Training

```bash
python bbg_Bigram.py
```

## 4. Generate Text

Generation runs automatically after training completes.

---

# Why This Project Matters

Many AI projects rely only on calling pretrained APIs. This project demonstrates direct understanding of the architecture powering modern LLM systems by implementing the components manually.

It shows practical knowledge of deep learning fundamentals, not just tool usage.

---

# Credits

Inspired by Andrej Karpathy's educational nanoGPT material.

Implemented independently as a learning and portfolio project.

---

# License

This project is available under the MIT License.
