# Transformer Model: Answer to messages on discord

Now I'll leave an explanation on the math behind the code, puhuhu..
---

## 1. Notation

* \$d\_{model}\$: dimensionality of embeddings and hidden states
* \$L\$: maximum sequence length
* \$N\$: vocabulary size
* \$X = \[x\_1, x\_2, \dots, x\_T]\$: input token sequence, each \$x\_t \in {1,\dots,N}\$
* \$\mathbf{E}(x) \in \mathbb{R}^{d\_{model}}\$: embedding of token \$x\$
* \$\mathbf{W}\_q,\mathbf{W}*k,\mathbf{W}*v \in \mathbb{R}^{d*{model}\times d*{model}}\$: learned projections for queries, keys, and values

---

## 2. Positional Encoding

**Math** For position \$p=0,1,\dots,L-1\$ and dimension index \$i=0,1,\dots,d\_{model}-1\$:

$$
\mathrm{PE}(p,2i)   = \sin\bigl(\frac{p}{10000^{2i/d_{model}}}\bigr),
\\
\mathrm{PE}(p,2i+1) = \cos\bigl(\frac{p}{10000^{2i/d_{model}}}\bigr)
$$

We add this deterministic matrix to token embeddings: \$\mathbf{z}\_t = \mathbf{E}(x\_t) + \mathrm{PE}(t)\$

```python
class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__();
        pe = torch.zeros(max_len, d_model);  # tensor of zeros for position encodings
        position = torch.arange(max_len).float().unsqueeze(1);  # positions [0..L-1]
        embedding_index = torch.arange(0, d_model, 2).float();  # even indices
        div_term = 1 / (10000.0 ** (embedding_index / d_model));  # scale term
        pe[:, 0::2] = torch.sin(position * div_term);
        pe[:, 1::2] = torch.cos(position * div_term);
        self.register_buffer("pe", pe);

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[: word_embeddings.size(0), :];
```

---

## 3. Scaled Dot-Product Self-Attention

**Math Steps**

1. **Projections**

   $$
   \mathbf{Q} = \mathbf{W}_q \mathbf{Z},\quad
   \mathbf{K} = \mathbf{W}_k \mathbf{Z},\quad
   \mathbf{V} = \mathbf{W}_v \mathbf{Z}
   $$

2. **Similarity matrix**

   $$
   \mathbf{S} = \mathbf{Q} \mathbf{K}^T,\quad
   \mathbf{S}_{scaled} = \frac{\mathbf{S}}{\sqrt{d_{model}}}
   $$

3. **Causal mask** prevents attending to future tokens using a lower-triangular mask

4. **Softmax**

   $$
   \mathbf{A} = \mathrm{softmax}(\mathbf{S}_{scaled} + M)
   $$

5. **Weighted sum**

   $$
   \mathrm{Attention}(\mathbf{Z}) = \mathbf{A} \mathbf{V}
   $$

```python
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__();
        self.W_q = nn.Linear(d_model, d_model, bias=False);
        self.W_k = nn.Linear(d_model, d_model, bias=False);
        self.W_v = nn.Linear(d_model, d_model, bias=False);
        self.row_dim = 0  # query dimension
        self.col_dim = 1  # key dimension

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q);
        k = self.W_k(encodings_for_k);
        v = self.W_v(encodings_for_v);
        sims = torch.matmul(q, k.transpose(self.row_dim, self.col_dim));  # S = Q K^T
        scaled_sims = sims / torch.sqrt(torch.tensor(k.size(self.col_dim), dtype=torch.float));
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, -1e9)
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)
        return attention_scores;
```

---

## 4. Decoder-Only Transformer Architecture

**Pipeline per token sequence**

1. **Embedding + Positional Encoding**

   $$
   \mathbf{Z} = \mathbf{E}(X) + \mathrm{PE}
   $$
2. **Self-Attention** with causal mask

   $$
   \mathbf{H} = \mathrm{Attention}(\mathbf{Z}, \mathbf{Z}, \mathbf{Z}, M)
   $$
3. **Residual connection**

   $$
   \mathbf{R} = \mathbf{Z} + \mathbf{H}
   $$
4. **Output layer**

   $$$
   \mathbf{Y} = \mathbf{R} \mathbf{W}_{out} + b_{out}
   $$ projecting to vocabulary logits
   $$$

```python
class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__();
        L.seed_everything(seed=42);
        self.we = nn.Embedding(num_tokens, d_model);
        self.pe = PositionEncoding(d_model, max_len);
        self.self_attention = Attention(d_model);
        self.fc_layer = nn.Linear(d_model, num_tokens);
        self.loss = nn.CrossEntropyLoss();

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids);
        position_encoded = self.pe(word_embeddings);
        mask = torch.tril(torch.ones((token_ids.size(0), token_ids.size(0)))).bool();
        mask = mask == 0;
        self_attention_values = self.self_attention(
            position_encoded, position_encoded, position_encoded, mask=mask
        );
        residual_connection_values = position_encoded + self_attention_values;
        fc_layer_output = self.fc_layer(residual_connection_values);
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
\mathcal{L} = -\sum_{t=1}^T \log P(y_t \mid x_{<t})
$$

implemented via:

```python
self.loss = nn.CrossEntropyLoss()
```

Optimizer:

```python
Adam(self.parameters(), lr=0.1)
```

---

## 6. Inference (Greedy Decoding)

1. Append `<EOS>` to the prompt
2. Loop until `<EOS>` or max length

   * Compute logits: `predictions = model(model_input)`
   * Choose next token: `predicted_id = torch.argmax(predictions[-1, :])`
   * Break if `<EOS>`
   * Append and repeat

```python
def generate(model, prompt, max_length=6) -> str:
    tokens = torch.tensor([token_to_id[t] for t in prompt.split()] + [token_to_id['<EOS>']]);
    predicted_ids = tokens.clone();
    for _ in range(tokens.size(0), max_length):
        predictions = model(predicted_ids);
        predicted_id = torch.tensor([torch.argmax(predictions[-1, :])]);
        if predicted_id == token_to_id['<EOS>']:
            break;
        predicted_ids = torch.cat((predicted_ids, predicted_id));
    return ''.join([id_to_token[i.item()] for i in predicted_ids if i != token_to_id['<EOS>']]);
```

---

**Reference** Vaswani et al., "Attention Is All You Need"
