import torch
import torch.nn as nn
import json
import re
import numpy as np
from flask import Flask, request, jsonify

# Load model config and vocab
with open('model_artifacts/vocabulary.json') as f:
    vocab = json.load(f)

inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocess and tokenize
def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    return text.lower().strip()

def tokenize(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

def decode(tokens):
    return ' '.join([inv_vocab.get(t, '<UNK>') for t in tokens if t not in [vocab['<PAD>'], vocab['<EOS>'], vocab['<SOS>']]])

def pad_sequence(seq, max_len, pad_val=0):
    return torch.tensor(seq + [pad_val] * (max_len - len(seq)), dtype=torch.long).unsqueeze(0).to(device)

# Define PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # Fixed: _init_ to __init__
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Define TransformerQA class
class TransformerQA(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dropout=0.1):  # Fixed: _init_ to __init__
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.embedding(src))
        tgt_emb = self.pos_encoder(self.embedding(tgt))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
        return self.fc(output)

# Load trained model
model = TransformerQA(vocab_size).to(device)
model.load_state_dict(torch.load('model_artifacts/transformer_qa_final.pth', map_location=device))
model.eval()

# Generate response function
def generate_response(query, max_len=50):
    query = preprocess(query)
    query_ids = tokenize(query)
    query_tensor = pad_sequence(query_ids, len(query_ids))

    src_mask = (query_tensor == vocab['<PAD>'])

    generated = [vocab['<SOS>']]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask = torch.triu(torch.full((len(generated), len(generated)), float('-inf')), diagonal=1).to(device)
        with torch.no_grad():
            out = model(query_tensor, tgt_tensor, src_key_padding_mask=src_mask, tgt_mask=tgt_mask)
            next_token = out[0, -1, :].argmax().item()

        if next_token == vocab['<EOS>']:
            break
        generated.append(next_token)

    return decode(generated)

# Initialize Flask app
app = Flask(__name__)

# API endpoint for query processing
@app.route('/predict', methods=['POST'])
def process_query():
    try:
        # Get JSON data from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Invalid query"}), 400

        # Generate response
        response = generate_response(query)
        
        # Return the response as JSON
        return jsonify({"response": response}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)