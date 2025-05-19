import pip;
import torch;
import torch.nn as nn;
import torch.nn.functional as F;
from torch.optim import Adam;
from torch.utils.data import TensorDataset, DataLoader;
import websocket;
import json;
import threading;
import time;
import requests;
with open("config.json") as f: token = json.load(f)['token'];
try:
    __import__("lightning");
except ImportError:
    pip.main(["install", "lightning"]);
import lightning as L;
token_to_id = {
    'what': 0,
    'is': 1,
    'sesso': 2,
    'verde': 3,
    '<EOS>': 4
};
id_to_token = dict(map(reversed, token_to_id.items()));
inputs = torch.tensor([[
    token_to_id["what"],
    token_to_id["is"],
    token_to_id["sesso"],
    token_to_id["<EOS>"],
    token_to_id["verde"]],
    
    [token_to_id["sesso"],
    token_to_id["is"],
    token_to_id["what"],
    token_to_id["<EOS>"],
    token_to_id["verde"]]]
);
labels = torch.tensor([[
    token_to_id["is"],
    token_to_id["sesso"],
    token_to_id["<EOS>"],
    token_to_id["verde"],
    token_to_id["<EOS>"]],
    
    [token_to_id["is"],
    token_to_id["what"],
    token_to_id["<EOS>"],
    token_to_id["verde"],
    token_to_id["<EOS>"]]]
);
dataset = TensorDataset(inputs, labels);
dataloader = DataLoader(dataset);
class PositionEncoding(nn.Module):
    def __init__(self, d_model=2, max_len=6):
        super().__init__();
        pe = torch.zeros(max_len, d_model);  # tensor of zeros for the position encodings
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1);  # Position vector: p = [0, 1, 2, ..., max_len-1]
        embedding_index = torch.arange(start=0, end=d_model, step=2).float();  # even positions in the encoding
        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model);  # scale term: div_term = 1 / (10000^(i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term);  # sin function to even indices: PE(p) = sin(p / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term);  # cos function to odd indices: PE(p) = cos(p / (10000^(2i/d_model)))
        self.register_buffer("pe", pe);  # register position encoding as a buffer
    def forward(self, word_embeddings):
        # position encoding to word embeddings: E(t) = word_embedding(t) + PE(t)
        return word_embeddings + self.pe[:word_embeddings.size(0), :];
class Attention(nn.Module):
    def __init__(self, d_model=2):
        super().__init__();
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False);  # query transformation: Q = W_q * X
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False);  # key transformation: K = W_k * X
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False);  # value transformation: V = W_v * X
        self.row_dim = 0;  # row dimension for matrix multiplication (query dimension)
        self.col_dim = 1;  # column dimension for matrix multiplication (key dimension)
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        q = self.W_q(encodings_for_q);  # query transformation: Q = W_q * encodings_for_q
        k = self.W_k(encodings_for_k);  # key transformation: K = W_k * encodings_for_k
        v = self.W_v(encodings_for_v);  # value transformation: V = W_v * encodings_for_v
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim));  # similarity: S = Q * K^T
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5);  # scale similarity: S_scaled = S / sqrt(d_k), where d_k is the key dimension
        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9);  # set invalid positions to minus infinity but we don't have minus infinity in python -1e9 works too
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim);  # softmax to get attention scores: α = softmax(S_scaled) owo
        attention_scores = torch.matmul(attention_percents, v);  # weighted sum of values: A = α * V
        return attention_scores;
class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=4, d_model=2, max_len=6):
        super().__init__();
        L.seed_everything(seed=42);
        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model);  # word embedding: E = W(t)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len);  # position Encoding: PE(t) = f(t, pos)
        self.self_attention = Attention(d_model=d_model);  # self-attention: Attention(Q, K, V)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens);  # fully connected layer: FC(x) = W*x + b
        self.loss = nn.CrossEntropyLoss();  # cross entropy loss: L = -∑(y * log(p))
    def forward(self, token_ids):
        word_embeddings = self.we(token_ids);  # word Embedding: W(t) → E(t)
        position_encoded = self.pe(word_embeddings);  # position Encoding: PE(E(t))
        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0)))).bool();  # mask: M(i, j) = 1 if i <= j else 0
        mask = mask == 0;  # mask: M(i, j) = 0 where i > j
        self_attention_values = self.self_attention(  # self-Attention: Attention(Q, K, V)
            position_encoded,
            position_encoded,
            position_encoded,
            mask=mask
        );
        residual_connection_values = position_encoded + self_attention_values;  # residual connection: R = X + Attention(Q, K, V)
        fc_layer_output = self.fc_layer(residual_connection_values);  # linear transformation: Y = W*R + b
        return fc_layer_output;
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1);  # adam Optimization: θ = θ - α * ∇θJ(θ)
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch;
        output = self.forward(input_tokens[0]);  # forward pass: output = f(input)
        loss = self.loss(output, labels[0]);  # cross entropy loss: L = -∑(y * log(p))
        return loss;
def run_transformer_model(training_mode:bool=False, channel_id:int=None, sender:bool=False, x:str="", epoch:int=30) -> str:
    x = x+" <EOS>";
    model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6);
    if training_mode:
        trainer = L.Trainer(max_epochs=epoch);
        trainer.fit(model, train_dataloaders=dataloader);
        torch.save(model.state_dict(), "model.pth");
    try:
        model.load_state_dict(torch.load("model.pth")); model.eval();
    except: pass;
    model_input = torch.tensor([token_to_id[token] for token in x.split()]);
    input_length = model_input.size(dim=0);
    predictions = model(model_input);
    predicted_id = torch.tensor([torch.argmax(predictions[-1,:])]);
    predicted_ids = predicted_id;
    max_length:int = 6;
    for i in range(input_length, max_length):
        if (predicted_id == token_to_id["<EOS>"]):
            break;
        model_input = torch.cat((model_input, predicted_id));
        predictions = model(model_input);
        predicted_id = torch.tensor([torch.argmax(predictions[-1,:])]);
        predicted_ids = torch.cat((predicted_ids, predicted_id));
    if sender:
        requests.post("https://discord.com/api/v9/channels/%s/messages" % channel_id, headers={
            "Authorization": token
        }, json={
            "content": "".join([str(id_to_token[i.item()]) for i in predicted_ids]).replace("<EOS>", "")
        }); return;
    print("Tokens:\n")
    for id in predicted_ids:
        print("\t", str(id_to_token[id.item()]));
def send_json_request(ws, request): ws.send(json.dumps(request));
def receive_json_response(ws):
    response = ws.recv();
    if response: return json.loads(response);
def heartbeat(interval, ws):
    while True:
        time.sleep(interval);
        heartbeatJSON = {
            "op": 1,
            "d": "null"
        };
        send_json_request(ws, heartbeatJSON);
def messagelogger():
    ws = websocket.WebSocket();
    ws.connect("wss://gateway.discord.gg/?v=6&encording=json");
    event = receive_json_response(ws);
    heartbeat_interval = event['d']['heartbeat_interval']/1000;
    threading._start_new_thread(heartbeat, (heartbeat_interval, ws));
    payload = {
        "op": 2,
        "d": {
            "token": token,
            "properties": {
                "$os": "windows",
                "$browser": "chrome",
                "$device": "pc"
            }
        }
    };
    send_json_request(ws, payload); print("Ready.");
    while True:
        event = receive_json_response(ws);
        try:
            chan_id:str = str(event['d']['channel_id']);
            _input = str(event['d']['content']).lower(); xxx = str(event['d']['author']['id']);
            print(_input);
            sender_user_id = str(__import__("base64").b64decode(token.split(".")[0]+"==").decode("utf-8"));
            for _ in list(token_to_id):
                if _ in _input and len(_input.split())==3 and not xxx==sender_user_id:
                    run_transformer_model(training_mode=False, sender=True, channel_id=chan_id, x=_input);
                    break;
        except Exception as y:
            print(y);
threading.Thread(target=messagelogger).start();