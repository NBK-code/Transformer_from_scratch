# Transformer From Scratch (PyTorch)

This project implements the **Transformer architecture from scratch in PyTorch**, following the original paper  
**“Attention Is All You Need” (Vaswani et al., 2017)** — no use of `nn.Transformer` or any high-level wrappers.

It includes a **complete encoder-decoder model**, custom **multi-head attention**, **positional encodings**, **feed-forward layers**, and **layer normalization**, along with dataset preprocessing, training loop, and evaluation metrics.

We train the model to translate from English to French using the Hugginface Opus Books dataset.

---

## Features

-  **Pure PyTorch implementation** of the full Transformer
-  Modular design — each block (Encoder, Decoder, Attention, etc.) is a separate class
-  Works with bilingual translation datasets via Hugging Face Datasets
-  Custom **multi-head self-attention** and **causal masking**
-  Implements **Layer Normalization**, **Residual Connections**, and **Positional Encodings**
-  Training with checkpoint saving and TensorBoard logging
-  Validation using **BLEU**, **CER**, and **WER** metrics

## Project Structure
```bash
Transformer_from_scratch/
│
├── model.py 
├── dataset.py 
├── train.py 
├── config.py 
├── requirements.txt 
├── README.md 
└── saved_weights
```
---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Transformer_from_scratch.git
cd Transformer_from_scratch
pip install -r requirements.txt
```

## Training

Run training with:
```bash
python train.py
```

This will:

- Load a bilingual dataset (e.g., English–French Opus books dataset from Hugging Face)
- Build tokenizers for source & target languages
- Train the Transformer model from scratch
- Save model checkpoints per epoch
- Log progress and metrics to TensorBoard

## Inference Example
To perform greedy decoding on a sentence (translate):
```bash
from model import build_transformer
from train import greedy_decode

output_tokens = greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device)
translation = tokenizer_tgt.decode(output_tokens.tolist())
print(translation)
```

## Theory Recap

The model follows the standard Transformer encoder-decoder architecture. Both encoder and decoder are built from N stacked blocks.

Each block includes:
- Multi-head attention
- Feed-forward network
- Residual connections
- Layer normalization


### Understanding the Attention Mechanism

The **core of the Transformer** is the **attention mechanism**, which allows each token in a sequence to dynamically focus on other relevant tokens.


<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V" alt="Attention">
</p>

Here:
- \(Q\): Query matrix of shape (n<sub>q</sub>, d<sub>k</sub>)  
- \(K\): Key matrix of shape (n<sub>k</sub>, d<sub>k</sub>)  
- \(V\): Value matrix of shape (n<sub>k</sub>, d<sub>v</sub>)  
- \(d<sub>k</sub>\): Dimension of the key/query vectors  

The output is a matrix of shape **(n<sub>q</sub>, d<sub>v</sub>)** — one output vector per query.


Although the formula uses all vectors at once, we can focus on **a single query vector** \(q<sub>i</sub>\).

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\mathrm{Attention}(q_i,K,V)=\sum_{j=1}^{n_k}\alpha_{ij}v_j" alt="Single attention equation">
</p>

Each attention weight \(α<sub>ij</sub>\) is computed as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\alpha_{ij}=\frac{\exp\left(\frac{q_i\cdot k_j}{\sqrt{d_k}}\right)}{\sum_{l=1}^{n_k}\exp\left(\frac{q_i\cdot k_l}{\sqrt{d_k}}\right)}" alt="Alpha weight equation">
</p>


### Step-by-Step Intuition

#### 1️⃣ Compute Similarities
We compute dot products between the query \(q<sub>i</sub>\) and all keys \(k<sub>j</sub>\):

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?s_j=q_i\cdot k_j" alt="Dot product similarity">
</p>

Each \(s<sub>j</sub>\) measures how similar the query is to key \(k<sub>j</sub>\).



#### 2️⃣ Scale and Normalize
We scale by (1 / √dₖ) for numerical stability, then apply **softmax** across all keys:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\alpha_{ij}=\mathrm{softmax}_j\left(\frac{s_j}{\sqrt{d_k}}\right)" alt="Softmax scaling">
</p>

These weights form a probability distribution — they sum to 1 across all keys.


#### 3️⃣ Weighted Sum of Values
Each output vector z<sub>i</sub> is a weighted sum of the values z<sub>j</sub>:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?z_i=\sum_j\alpha_{ij}v_j" alt="Weighted sum of values">
</p>

This means each query’s new representation z<sub>i</sub> is a **blend of the value vectors**, with weights depending on similarity to the keys.


### Summary

| Symbol | Meaning |
|---------|----------|
| q<sub>i</sub> | What we’re **looking for** (the query) |
| k<sub>j</sub> | What each token **represents** (the key) |
| v<sub>j</sub> | What each token **offers as information** (the value) |
| α<sub>ij</sub> | How much **attention** q<sub>i</sub> gives to v<sub>j</sub> |
| z<sub>i</sub> | The **new, context-aware representation** of \(q_i\) |

---

## References

1. [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762)
2. [Umar Jamil's GitHub Transformer repository](https://github.com/hkproj/pytorch-transformer/tree/main)
3. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
4. [Hugging Face Datasets](https://huggingface.co/docs/datasets)
