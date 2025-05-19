# Transformer Model: Answer to Messages on Discord

Now I'll leave an explanation on the math behind the code, puhuhu...

---

## 1. Notation

- `d_model`: Dimensionality of embeddings and hidden states (integer)
- `L`: Maximum sequence length (integer)
- `N`: Vocabulary size (integer)
- `X = [x₁, x₂, ..., x_T]`: Input token sequence (x_t ∈ {1,...,N})
- `E(x) ∈ ℝ^(d_model)`: Embedding of token x
- `W_q`, `W_k`, `W_v ∈ ℝ^(d_model×d_model)`: Learned projections for queries/keys/values

---

## 2. Positional Encoding

**Math** For position `p=0,1,...,L-1` and dimension index `i=0,1,...,d_model-1`:

```
PE(p,2i)   = sin(p / 10000^(2i/d_model))
PE(p,2i+1) = cos(p / 10000^(2i/d_model))
```

We add this to token embeddings:  
`zₜ = E(xₜ) + PE(t)`

*(Implementation code remains unchanged)*

---

## 3. Scaled Dot-Product Self-Attention

**Math Steps**

1. **Projections**  
   Q = W_q·Z  
   K = W_k·Z  
   V = W_v·Z

2. **Similarity Matrix**  
   S = Q·Kᵀ  
   S_scaled = S / √d_model

3. **Causal Mask**  
   Lower-triangular mask to prevent looking ahead

4. **Softmax**  
   A = softmax(S_scaled + M)

5. **Weighted Sum**  
   Attention(Z) = A·V

*(Implementation code remains unchanged)*

---

## 4. Decoder-Only Architecture

**Pipeline** (per token sequence):

1. **Embedding + PE**  
   Z = E(X) + PE

2. **Self-Attention**  
   H = Attention(Z, Z, Z, M)

3. **Residual Connection**  
   R = Z + H

4. **Output Layer**  
   Y = R·W_out + b_out  (→ vocabulary logits)

*(Implementation code remains unchanged)*

---

## 5. Training

**Loss**: Cross-entropy for next-token prediction  
```
L = -Σ log P(yₜ | x_<t)
```

**Optimizer**:  
```python
Adam(lr=0.1)
```

---

## 6. Inference (Greedy Decoding)

1. Append `<EOS>` to prompt
2. Loop until `<EOS>`/max_length:
   - Compute logits
   - Select argmax for next token
   - Break on `<EOS>`

*(Implementation code remains unchanged)*

---

## References

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
For proper math rendering:  
If you want to view this with proper LaTeX formatting, consider:
1. Using a browser extension like [MathJax for GitHub](https://github.com/orsharir/github-mathjax)
2. Viewing the README in a Markdown editor that supports LaTeX (Typora, Obsidian, etc.)
3. Converting to PDF with LaTeX support using pandoc
