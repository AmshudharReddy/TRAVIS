from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter
import math
import spacy
import io
import numpy as np
import traceback

# Initialize Flask app
app = Flask(__name__)

# --- Configuration & Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Model Hyperparameters
EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Global variables for model and transforms
model = None
vocab_transform = None
token_transform = None
text_transform = None

# --- Model Definition ---

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        seq_len = token_embedding.size(1)
        return self.dropout(token_embedding + self.pos_embedding[:, :seq_len, :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = SinusoidalPositionalEncoding(emb_size, dropout=dropout, maxlen=max_seq_len)

    def forward(self,
                src,
                trg,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))

        if src_mask is not None and src_mask.dtype != torch.bool: src_mask = src_mask.bool()
        if tgt_mask is not None and tgt_mask.dtype != torch.bool: tgt_mask = tgt_mask.bool()
        if src_padding_mask is not None and src_padding_mask.dtype != torch.bool: src_padding_mask = src_padding_mask.bool()
        if tgt_padding_mask is not None and tgt_padding_mask.dtype != torch.bool: tgt_padding_mask = tgt_padding_mask.bool()
        if memory_key_padding_mask is not None and memory_key_padding_mask.dtype != torch.bool: memory_key_padding_mask = memory_key_padding_mask.bool()

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask=None):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)),
                                        mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask=None, memory_key_padding_mask=None):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory,
                                        tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

# --- Helper Functions ---

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def tokenize_te_placeholder(text):
    return text.split()

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# --- Translation Function ---

def translate_sentence(src_sentence: str, max_len: int = 50):
    """Translate English sentence to Telugu"""
    global model, vocab_transform, text_transform
    
    model.eval()
    
    with torch.no_grad():
        # Preprocess the source sentence
        src_tensor = text_transform['en'](src_sentence).unsqueeze(0).to(DEVICE)
        src_padding_mask = (src_tensor == PAD_IDX).to(DEVICE)
        
        # Encoder output (memory)
        memory = model.encode(src_tensor, src_mask=None, src_padding_mask=src_padding_mask)
        memory = memory.to(DEVICE)
        
        # Start decoding with BOS token
        ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
        
        for _ in range(max_len - 1):
            tgt_seq_len = ys.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)
            
            out = model.decode(ys, memory, tgt_mask, 
                             tgt_padding_mask=None, 
                             memory_key_padding_mask=src_padding_mask)
            
            # Get the probability of the last token
            prob = model.generator(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()
            
            # Append predicted token to the sequence
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_idx)], dim=1)
            
            if next_word_idx == EOS_IDX:
                break
        
        # Convert token IDs to words
        tgt_tokens = [vocab_transform['te'].get_itos()[i] for i in ys.squeeze(0).tolist()]
        
        # Remove special tokens and join
        return " ".join([token for token in tgt_tokens if token not in special_symbols])

# --- Flask Routes ---
@app.route('/translate', methods=['POST'])
def translate():
    """Main translation endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            print("Error: Model not loaded")
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'success': False
            }), 500
        
        # Debug: Print request info
        print(f"Content-Type: {request.content_type}")
        print(f"Request data: {request.data}")
        
        # Get JSON data from request
        data = request.get_json()
        print(f"Parsed JSON: {data}")
        
        if not data:
            print("Error: No JSON data received")
            return jsonify({
                'error': 'No JSON data received. Please send request with Content-Type: application/json',
                'success': False
            }), 400
            
        if 'sentence' not in data:
            print("Error: Missing 'text' field")
            return jsonify({
                'error': 'Missing "text" field in request body. Expected format: {"text": "your text here"}',
                'success': False,
                'received_data': data
            }), 400
        
        english_text = data['sentence'].strip() if isinstance(data['sentence'], str) else str(data['sentence']).strip()
        print(f"Input text: '{english_text}'")
        
        if not english_text:
            print("Error: Empty text provided")
            return jsonify({
                'error': 'Empty text provided',
                'success': False
            }), 400
        
        # Optional parameters
        max_length = data.get('max_length', 50)
        print(f"Max length: {max_length}")
        
        # Perform translation
        print("Starting translation...")
        telugu_translation = translate_sentence(english_text, max_len=max_length)
        print(f"Translation result: '{telugu_translation}'")
        
        return jsonify({
            'success': True,
            'input': english_text,
            'translation': telugu_translation,
            'language_pair': 'en-te'
        })
        
    except Exception as e:
        print(f"Translation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Translation failed: {str(e)}',
            'success': False
        }), 500


def initialize_model():
    """Initialize and load the translation model"""
    global model, vocab_transform, token_transform, text_transform
    
    try:
        print("Loading vocabularies...")
        # Load vocabularies
        vocab_transform_en_loaded = torch.load('vocab_transform_en.pt', map_location=DEVICE)
        vocab_transform_te_loaded = torch.load('vocab_transform_te.pt', map_location=DEVICE)
        
        vocab_transform = {'en': vocab_transform_en_loaded, 'te': vocab_transform_te_loaded}
        SRC_VOCAB_SIZE = len(vocab_transform['en'])
        TGT_VOCAB_SIZE = len(vocab_transform['te'])
        
        print(f"Source (English) vocabulary size: {SRC_VOCAB_SIZE}")
        print(f"Target (Telugu) vocabulary size: {TGT_VOCAB_SIZE}")
        
        # Initialize tokenizers
        print("Initializing tokenizers...")
        try:
            nlp_en = spacy.load('en_core_web_sm')
        except OSError:
            raise Exception("Spacy 'en_core_web_sm' model not found. Please run: python -m spacy download en_core_web_sm")
        
        token_transform = {}
        token_transform['en'] = get_tokenizer('spacy', language='en_core_web_sm')
        token_transform['te'] = get_tokenizer(tokenize_te_placeholder)
        
        # Create text transforms
        text_transform = {}
        for ln in ['en', 'te']:
            text_transform[ln] = sequential_transforms(token_transform[ln],
                                                    vocab_transform[ln],
                                                    tensor_transform)
        
        # Initialize model
        print("Initializing model...")
        model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT)
        model = model.to(DEVICE)
        
        # Load trained model weights
        print("Loading model weights...")
        model_path = 'transformer_eng_tel_scratch_full_data.pt'
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        print("Model loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure all required files exist:")
        print("- vocab_transform_en.pt")
        print("- vocab_transform_te.pt") 
        print("- transformer_eng_tel_scratch_full_data.pt")
        return False
    except Exception as e:
        print(f"Error initializing model: {e}")
        print(traceback.format_exc())
        return False

# --- Main Execution ---

if __name__ == '__main__':
    print("Initializing English to Telugu Translation API...")
    
    if initialize_model():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5002, debug=False)
    else:
        print("Failed to initialize model. Server not started.")
        exit(1)