# Transformer Attention, Online Softmax & KV-Caching

A Python/NumPy implementation demonstrating the algorithmic evolution of Transformer Attention — from a naive implementation of Softmax to numerically stable attention, online streaming, and KV-cached autoregressive generation.

This repository is our group's submission for **Problem 4** of the Algorithms and Data Structures examination.

---

## Overview

Large Language Models (LLMs) rely on the Attention mechanism to process sequences and understand relationships between tokens.

A standard Transformer uses **Scaled Dot-Product Attention**, which involves computing similarity scores between Query and Key vectors, applying Softmax, and combining the resulting probabilities with Value vectors.

Although the mathematical formula is straightforward, a naive implementation can introduce several challenges:

- Numerical overflow when calculating exponential values.
- High intermediate memory usage when storing complete score and probability matrices.
- Redundant computations during autoregressive text generation.

This project explores these challenges through a **5-Stage Algorithmic Ladder**, showing how each stage improves upon the previous approach.

### Main Improvements

1. **Numerical Stability:** Preventing exponential overflow using scaling and Max-Shifting (Stage 1 → Stage 2).
2. **Memory Efficiency:** Processing Softmax incrementally using Online Streaming Softmax (Stage 2 → Stage 3).
3. **Computational Efficiency:** Avoiding repeated calculations during autoregressive generation using KV-Caching (Stage 4 → Stage 5).

---

## The 5-Stage Evolution

### Stage 1: Naive Unscaled Dot-Product Attention & Textbook Softmax

**Concept:**

Directly implements the basic Attention formula:

\[
O = \operatorname{softmax}(QK^T)V
\]

The implementation:
- Computes the full Query-Key score matrix.
- Applies a causal mask to prevent access to future tokens.
- Calculates Softmax using the standard exponential formula.
- Multiplies the probability matrix by the Value matrix.

**Limitation:**

Large score values can cause exponential overflow when using `float32` arithmetic.

For example, the maximum finite `float32` value is approximately \(3.4 \times 10^{38}\). Exponential values can exceed this limit when their input is approximately greater than 88.7.

This may result in `inf` and, subsequently, `NaN` values during Softmax calculations.

---

### Stage 2: Variance-Scaled Two-Pass Safe Softmax

**Concept:**

Improves numerical stability by introducing two techniques:

1. **Scaling:** Divides the Query-Key dot products by the square root of the Key dimension.

\[
S = \frac{QK^T}{\sqrt{d_k}}
\]

2. **Max-Shifting:** Subtracts the maximum score from each row before calculating exponentials.

\[
S'_{ij} = S_{ij} - \max_j(S_{ij})
\]

**Improvement:**

After Max-Shifting, the largest value in each row becomes zero:

\[
e^0 = 1
\]

This prevents exponential overflow for finite input scores and makes the Softmax calculation numerically stable.

The implementation still stores the complete score and probability matrices.

---

### Stage 3: Single-Pass Online Streaming Softmax

**Concept:**

Replaces the conventional Softmax calculation with an online streaming approach.

Instead of requiring the complete set of scores before calculating Softmax, the algorithm processes scores incrementally while maintaining:

- A running maximum (\(m_k\)).
- A running denominator (\(d_k\)).
- A correction factor to adjust previous accumulated values when the running maximum changes.

The running maximum is updated as follows:

\[
m_k = \max(m_{k-1}, x_k)
\]

The denominator is updated using:

\[
d_k =
d_{k-1}e^{m_{k-1}-m_k}
+
e^{x_k-m_k}
\]

**Improvement:**

Online Softmax avoids the need to store the complete intermediate score and probability matrices.

It reduces intermediate memory requirements and enables streaming computation, which is especially useful for long sequences.

---

### Stage 4: Naive Causal Autoregressive Generation Loop

**Concept:**

Implements autoregressive text generation without using a KV-Cache.

At every generation step, the algorithm:

1. Converts the entire current token sequence into embeddings.
2. Recomputes the Query, Key, and Value matrices for all tokens.
3. Recomputes the attention scores and causal Softmax.
4. Calculates the context vectors.
5. Predicts the next token and appends it to the sequence.

**Limitation:**

Previously processed tokens are repeatedly evaluated at every generation step.

As the sequence grows, the amount of repeated computation increases significantly.

For \(T\) generation steps, the total attention matrix multiplication work is:

\[
2d_k\sum_{t=1}^{T}t^2
\]

Therefore, the overall attention computation has cubic time complexity:

\[
\boxed{\Theta(T^3d_k)}
\]

This makes the naive approach increasingly inefficient for long sequences.

---

### Stage 5: Pre-Allocated Key-Value (KV) Cache

**Concept:**

Introduces a pre-allocated memory structure for storing previously computed Key and Value vectors.

Instead of recomputing Keys and Values for all previous tokens, the algorithm:

1. Computes the Query, Key, and Value vectors for the new token.
2. Stores the new Key and Value vectors in their respective buffers.
3. Reuses the previously cached Keys and Values.
4. Calculates attention using the available cached sequence.
5. Predicts the next token.

**Improvement:**

KV-Caching eliminates the need to recompute previous Key and Value projections during autoregressive decoding.

The total attention matrix multiplication work becomes:

\[
2d_k\sum_{t=1}^{T}t
\]

Therefore, the overall attention computation is reduced to:

\[
\boxed{\Theta(T^2d_k)}
\]

The cache requires additional memory to store the Key and Value vectors:

\[
\boxed{\Theta(Td_k)}
\]

This approach follows a key optimization used in modern LLM inference systems.

---

## Complexity Comparison

The following table compares the main algorithmic characteristics of the five stages.

| Stage | Main Approach | Attention Time Complexity | Intermediate / Cache Space |
|---|---|---:|---:|
| Stage 1 | Naive Softmax | \(\Theta(n^2d_k)\) | \(\Theta(n^2)\) |
| Stage 2 | Scaled Safe Softmax | \(\Theta(n^2d_k)\) | \(\Theta(n^2)\) |
| Stage 3 | Online Streaming Softmax | \(\Theta(n^2d_k)\) | Reduced intermediate memory |
| Stage 4 | Generation without Cache | \(\Theta(T^3d_k)\) | \(\Theta(T^2)\) |
| Stage 5 | Generation with KV-Cache | \(\Theta(T^2d_k)\) | \(\Theta(Td_k)\) |

**Note:** The Stage 4 and Stage 5 time complexities describe the main attention matrix multiplications during generation. Other operations, such as projection and output-layer computations, are not included in the multiplication counter used for this comparison.

---

## Project Layout

```text
├── stage1_2_safe_attention.py
│   └── Implementations of Stage 1 & Stage 2
│
├── stage3_online_softmax.py
│   └── Implementation of Online Streaming Softmax
│
├── stage4_naive_generation.py
│   └── Autoregressive generation without KV-Cache
│
├── stage5_kvcache_generation.py
│   └── Autoregressive generation with KV-Cache
│
└── README.md
    └── Project documentation
```

---

## Requirements

The project uses Python and NumPy.

- Python 3.x
- NumPy

Install NumPy using:

```bash
pip install numpy
```

---

## How to Run

Run each Python implementation independently:

```bash
python stage1_2_safe_attention.py
```

```bash
python stage3_online_softmax.py
```

```bash
python stage4_naive_generation.py
```

```bash
python stage5_kvcache_generation.py
```

Each script contains its corresponding implementation and test code.

---

## Conclusion

This project demonstrates how the Attention mechanism can be improved through a sequence of algorithmic optimizations.

- **Stage 1 → Stage 2:** Improves numerical stability.
- **Stage 2 → Stage 3:** Reduces intermediate memory requirements through online computation.
- **Stage 4 → Stage 5:** Reduces redundant computation during autoregressive generation through KV-Caching.

Together, these stages illustrate the importance of numerical stability, memory efficiency, and time complexity when implementing Transformer Attention.

---

**Course:** Algorithms and Data Structures  
**Assignment:** Problem 4  
**Project:** Transformer Attention, Online Softmax & KV-Caching