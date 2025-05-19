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

```python
class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # Tensor of zeros for position encodings
        position = torch.arange(max_len).float().unsqueeze(1)  # Positions [0..L-1]
        embedding_index = torch.arange(0, d_model, 2).float()  # Even indices
        div_term = 1 / (10000.0 ** (embedding_index / d_model))  # Scale term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[: word_embeddings.size(0), :]
```
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

```python
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_emb, k_emb, v_emb, mask=None):
        Q = self.W_q(q_emb)
        K = self.W_k(k_emb)
        V = self.W_v(v_emb)
        
        sims = Q @ K.transpose(0, 1)
        scaled_sims = sims / (K.size(-1) ** 0.5)
        
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, -1e9)
            
        attn = F.softmax(scaled_sims, dim=-1)
        return attn @ V
```

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

```python
class DecoderTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_model)
        self.pe = PositionEncoding(d_model, max_len)
        self.attn = Attention(d_model)
        self.out = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tokens):
        # Shape: (seq_len, d_model)
        embeds = self.pe(self.embed(tokens))
        
        # Causal mask
        mask = ~torch.tril(torch.ones(len(tokens), len(tokens))).bool()
        
        # Self-attention
        attn_out = self.attn(embeds, embeds, embeds, mask)
        residual = embeds + attn_out
        return self.out(residual)

    def training_step(self, batch):
        x, y = batch
        preds = self(x)
        return self.loss_fn(preds.view(-1, preds.size(-1)), y.view(-1))
```

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

```python
def generate(model, prompt, max_len=10):
    tokens = [token_to_id[p] for p in prompt.split()]
    while len(tokens) < max_len:
        logits = model(torch.tensor(tokens))
        next_id = logits[-1].argmax().item()
        if next_id == EOS_ID: break
        tokens.append(next_id)
    return ' '.join([id_to_token[t] for t in tokens])
```
---

## References

Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
