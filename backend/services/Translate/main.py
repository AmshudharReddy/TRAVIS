import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchtext.data.utils import get_tokenizer
from flask import Flask, request, jsonify

# Define the Embeddings class
class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, d_model)
        for k in range(max_length):
            for i in range(d_model // 2):
                theta = k / (10000 ** ((2 * i) / d_model))
                pe[k, 2 * i] = math.sin(theta)
                pe[k, 2 * i + 1] = math.cos(theta)
        # pe = pe.unsqueeze(0)  # Add batch dimension: [1, max_length, d_model]
        self.register_buffer("pe", pe)  # Register the tensor as a buffer
        # Shape: [1, max_length, d_model] for broadcasting

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)  # Shape: [1, seq_len, d_model]
        return self.dropout(x)

# Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_key).permute(0, 2, 1, 3)
        scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_key)
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)
        attn_probs = torch.softmax(scaled_dot_prod, dim=-1)
        A = torch.matmul(self.dropout(attn_probs), V)
        A = A.permute(0, 2, 1, 3).contiguous()
        A = A.view(batch_size, -1, self.n_heads * self.d_key)
        output = self.Wo(A)
        return output, attn_probs

# Define the PositionwiseFeedForward class
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear_layer_1 = nn.Linear(d_model, d_ffn)
        self.linear_layer_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.linear_layer_2(self.dropout(F.relu(self.linear_layer_1(x))))

# Define the EncoderLayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.positionwise_fnn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.fnn_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, attn_probs = self.attention(src, src, src, src_mask)
        src = self.attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_fnn(src)
        src = self.fnn_layer_norm(src + self.dropout(_src))
        return src, attn_probs

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        for layer in self.layers:
            src, _ = layer(src, src_mask)
        return src

# Define the DecoderLayer class
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int, dropout: float):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.masked_attn_layer_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.positionwise_fnn = PositionwiseFeedForward(d_model, d_ffn, dropout)
        self.fnn_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        _trg, _ = self.masked_attention(trg, trg, trg, trg_mask)
        trg = self.masked_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attn_probs = self.attention(trg, src, src, src_mask)
        trg = self.attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_fnn(trg)
        trg = self.fnn_layer_norm(trg + self.dropout(_trg))
        return trg, attn_probs

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.Wo = nn.Linear(d_model, vocab_size)

    def forward(self, trg, src, trg_mask, src_mask):
        for layer in self.layers:
            trg, _ = layer(trg, src, trg_mask, src_mask)
        return self.Wo(trg)

# Define the Transformer class
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Module, trg_embed: nn.Module, src_pad_idx: int, trg_pad_idx: int, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        seq_length = trg.shape[1]
        trg_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device)).bool()
        return trg_mask & trg_sub_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        src = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.trg_embed(trg), src, trg_mask, src_mask)
        return output

# Function to create the model
def make_model(device, src_vocab, trg_vocab, n_layers=3, d_model=256, d_ffn=512, n_heads=8, dropout=0.1, max_length=50):
    encoder = Encoder(d_model, n_layers, n_heads, d_ffn, dropout)
    decoder = Decoder(len(trg_vocab), d_model, n_layers, n_heads, d_ffn, dropout)
    src_embed = nn.Sequential(Embeddings(len(src_vocab), d_model), PositionalEncoding(d_model, dropout, max_length))
    trg_embed = nn.Sequential(Embeddings(len(trg_vocab), d_model), PositionalEncoding(d_model, dropout, max_length))
    model = Transformer(encoder, decoder, src_embed, trg_embed, src_vocab.get_stoi()["<pad>"], trg_vocab.get_stoi()["<pad>"], device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# Load vocabularies (assumes vocab_src.pt and vocab_trg.pt are saved from training)
vocab_src = torch.load("en_vocab.pt")
vocab_trg = torch.load("te_vocab.pt")

# Initialize Flask app
app = Flask(__name__)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model with the same architecture as during training
model = make_model(device, vocab_src, vocab_trg, n_layers=3, d_model=256, d_ffn=512, n_heads=8, dropout=0.1, max_length=50)

# Load the trained model weights
model.load_state_dict(torch.load("transformer-model_tel.pt", map_location=device))
model.to(device)
model.eval()

# Define the tokenizer for English
tokenizer_en = get_tokenizer("basic_english")

# Translation function
def translate_sentence(sentence, model, device, vocab_src, vocab_trg, tokenizer, max_length=50):
    model.eval()
    tokens = tokenizer(sentence)
    src = ['<bos>'] + [token.lower() for token in tokens] + ['<eos>']
    src_indexes = [vocab_src[token] if token in vocab_src else vocab_src['<unk>'] for token in src]
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)
    trg_indexes = [vocab_trg['<bos>']]
    for i in range(max_length):
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
            pred_token = output.argmax(2)[:, -1].item()
            if pred_token == vocab_trg['<eos>'] or i == max_length - 1:
                break
            trg_indexes.append(pred_token)
    trg_tokens = [vocab_trg.get_itos()[index] for index in trg_indexes[1:]]  # Skip '<bos>'
    return " ".join(trg_tokens)

@app.route('/translate',methods=['POST'])
def translate():
    try:
        data=request.get_json()
        if not data or 'sentence' not in data:
            return jsonify({"error":"No sentence provided"}),400
        
        sentence = data['sentence']
        if not isinstance(sentence, str) or not sentence.strip():
            return jsonify({"error": "Invalid sentence"}), 400
        
        translation = translate_sentence(sentence, model, device, vocab_src, vocab_trg, tokenizer_en)

        # Return the translation as JSON
        return jsonify({"translation": translation}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main execution
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5002,debug=True)