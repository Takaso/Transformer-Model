# Transformer Model: Answer to messages on discord

Now I'll leave an explanation on the math behind the code, puhuhu..

---

## 1. Notation

* $d_{model}$: dimensionality of embeddings and hidden states
* $L$: maximum sequence length
* $N$: vocabulary size
* $X = [x_1, x_2, \dots, x_T]$: input token sequence, each $x_t \in \{1,\dots,N\}$
* $\mathbf{E}(x)\in\mathbb{R}^{d_{model}}$: embedding of token $x$
* $\mathbf{W}_q,\mathbf{W}_k,\mathbf{W}_v\in\mathbb{R}^{d_{model}\times d_{model}}$: learned projections for queries, keys, values

---

## 2. Positional Encoding

**Math** For position $p=0,1,\dots,L-1$ and dimension index $i=0,1,\dots,d_{model}-1$:

$$
\mathrm{PE}(p,2i)   = \sin\bigl(p / 10000^{2i/d_{model}}\bigr)
\\
\mathrm{PE}(p,2i+1) = \cos\bigl(p / 10000^{2i/d_{model}}\bigr)
$$

We add this deterministic matrix to token embeddings: $\mathbf{z}_t = \mathbf{E}(x_t) + \mathbf{PE}[t]$

```python
class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__();
        pe = torch.zeros(max_len, d_model);  # tensor of zeros for position encodings
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1);  # positions [0..L-1]
        embedding_index = torch.arange(start=0, end=d_model, step=2).float();  # even indices
        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model);  # scale term
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term);
        self.register_buffer("pe", pe);  # register as buffer

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[: word_embeddings.size(0), :];
```

---

## 3. Scaled Dot-Product Self-Attention

**Math Steps**

1. **Projections**
   $\mathbf{Q} = \mathbf{W}_q\mathbf{Z},\quad
   \mathbf{K} = \mathbf{W}_k\mathbf{Z},\quad
   \mathbf{V} = \mathbf{W}_v\mathbf{Z}$

2. **Similarity matrix**
   $\mathbf{S} = \mathbf{Q}\mathbf{K}^T,\quad
   \mathbf{S}_{scaled} = \frac{\mathbf{S}}{\sqrt{d_{model}}}$

3. **Causal mask** prevents attending to future tokens using a lower-triangular mask

4. **Softmax**
   $\mathbf{A} = \mathrm{softmax}(\mathbf{S}_{scaled} + M)$

5. **Weighted sum**
   $\mathrm{Attention}(\mathbf{Z}) = \mathbf{A}\mathbf{V}$

```python
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__();
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False);
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False);
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False);
        self.row_dim = 0  # query dimension
        self.col_dim = 1  # key dimension

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q);  # Q = W_q X
        k = self.W_k(encodings_for_k);  # K = W_k X
        v = self.W_v(encodings_for_v);  # V = W_v X
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim));  # S = Q K^T
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5);  # scale by sqrt(d_k)
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)  # A = alpha V
        return attention_scores;
```

---

## 4. Decoder-Only Transformer Architecture

**Pipeline per token sequence**

1. **Embedding + Positional Encoding**
   $\mathbf{Z} = \mathbf{E}(X) + \mathbf{PE}$
2. **Self-Attention** with causal mask
   $\mathbf{H} = \mathrm{Attention}(\mathbf{Z},\mathbf{Z},\mathbf{Z}, M)$
3. **Residual connection**
   $\mathbf{R} = \mathbf{Z} + \mathbf{H}$
4. **Output layer**
   $\mathbf{Y} = \mathbf{R}\mathbf{W}_{out} + b_{out}$ projecting to vocabulary logits

```python
class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__();
        L.seed_everything(seed=42)
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model);
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len);
        self.self_attention = Attention(d_model=d_model);
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens);
        self.loss = nn.CrossEntropyLoss();

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids);  # E = W(t)
        position_encoded = self.pe(word_embeddings);  # + PE
        mask = torch.tril(torch.ones((token_ids.size(0), token_ids.size(0)))).bool()
        mask = mask == 0  # causal mask
        self_attention_values = self.self_attention(
            position_encoded, position_encoded, position_encoded, mask=mask
        )
        residual_connection_values = position_encoded + self_attention_values  # R = X + A
        fc_layer_output = self.fc_layer(residual_connection_values)  # logits
        return fc_layer_output;

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1);

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch;
        output = self.forward(input_tokens[0]);
        loss = self.loss(output, labels[0]);
        return loss;
```

---

## 5. Training and Loss

We train to predict the next token using **cross-entropy** loss

$$
\mathcal{L} = -\sum_{t=1}^T \log P(y_t \mid x_{<t}),
$$

implemented via

```python
self.loss = nn.CrossEntropyLoss()
```

Optimizer

```python
Adam(self.parameters(), lr=0.1)
```

---

## 6. Inference (Greedy Decoding)

1. Append `<EOS>` to the prompt
2. Loop until `<EOS>` or max length

   * Compute logits: `predictions = model(model_input)`
   * Choose next token: `predicted_id = torch.argmax(predictions[-1,:])`
   * Break if `<EOS>`
   * Append and repeat

```python
def generate(model, prompt, max_length=6) -> str:
    tokens = torch.tensor([token_to_id[t] for t in prompt.split()] + [token_to_id['<EOS>']])
    predicted_ids = tokens.clone()
    for _ in range(tokens.size(0), max_length):
        predictions = model(predicted_ids)
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])])
        if predicted_id == token_to_id['<EOS>']:
            break
        predicted_ids = torch.cat((predicted_ids, predicted_id))
    return ''.join([id_to_token[i.item()] for i in predicted_ids if i != token_to_id['<EOS>']])
```

---

## 7. Usage

```bash
# Train for 30 epochs
python train.py --epochs 30

# Generate text
python demo.py --prompt "what is sesso"
```

---

**Reference:** Vaswani et al., "Attention Is All You Need";
