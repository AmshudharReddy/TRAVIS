import torch
import torch.nn as nn
import json
import pickle
import re
import os
import numpy as np
from flask import Flask, request, jsonify

# Define paths (adjust these to relative paths if needed)
MODEL_PATH = 'transformer_qa_best.pth'
VOCAB_PATH = 'vocabulary.json'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
MODEL_CONFIG_PATH = 'model_config.json'

# Load model configuration
with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

# Load vocabulary
with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

# Load label encoder
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Define the TransformerClassifierQA class (same as before)
class TransformerClassifierQA(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2,
                 num_classes=1, max_answer_len=27):
        super(TransformerClassifierQA, self).__init__()
        self.d_model = d_model
        self.max_answer_len = max_answer_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos = nn.Parameter(torch.rand(1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder_pos = nn.Parameter(torch.rand(max_answer_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_encoder_layers)
        self.fc_class = nn.Linear(d_model, num_classes)
        self.fc_qa = nn.Linear(d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x) + self.encoder_pos[:x.size(1), :].unsqueeze(0)
        memory = self.encoder(x, src_key_padding_mask=attention_mask)
        batch_size = x.size(0)
        tgt = self.decoder_pos[:self.max_answer_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        decoded = self.decoder(tgt, memory, memory_key_padding_mask=attention_mask)
        cls_token = torch.mean(memory, dim=1)
        category_output = self.fc_class(cls_token)
        answer_output = self.fc_qa(decoded)
        return category_output, answer_output

# Initialize and load the model
model = TransformerClassifierQA(
    vocab_size=model_config['vocab_size'],
    d_model=model_config['d_model'],
    nhead=model_config['nhead'],
    num_encoder_layers=model_config['num_encoder_layers'],
    num_classes=model_config['num_classes'],
    max_answer_len=model_config['max_answer_len']
)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)  # Assuming the checkpoint is a direct state dict
model.eval()

# Preprocessing functions (same as before)
def preprocess(text):
    text = re.sub(r'\W', ' ', text).lower().strip()
    return text

def tokenize(text):
    return [vocab.get(word, vocab['<UNK>']) for word in text.split()]

def pad_sequence(sequence, max_len, padding_value=0):
    if len(sequence) > max_len:
        return sequence[:max_len]
    return sequence + [padding_value] * (max_len - len(sequence))

# Prediction function (same as before)
def predict(query):
    processed_query = preprocess(query)
    tokenized_query = tokenize(processed_query)
    padded_query = pad_sequence(tokenized_query, model_config['max_query_len'])
    input_tensor = torch.tensor([padded_query], dtype=torch.long)
    attention_mask = (input_tensor == 0)
    with torch.no_grad():
        category_output, answer_output = model(input_tensor, attention_mask=attention_mask)
    predicted_category = torch.argmax(category_output, dim=1).item()
    category_label = label_encoder.inverse_transform([predicted_category])[0]
    predicted_answer_tokens = torch.argmax(answer_output, dim=2).squeeze(0).tolist()
    answer_words = [word for word, idx in vocab.items() if idx in predicted_answer_tokens]
    response = ' '.join(answer_words)
    return category_label, response

# Initialize Flask app
app = Flask(__name__)

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    category, response = predict(query)
    return jsonify({'category': category, 'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)