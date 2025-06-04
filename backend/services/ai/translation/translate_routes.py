from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
import math
import spacy
import traceback
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create router for translation service
translation_router = APIRouter(prefix="/api", tags=["Translation"])

# Configuration & Hyperparameters
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

model = None
vocab_transform = None
token_transform = None
text_transform = None

class TranslationRequest(BaseModel):
    sentence: str
    max_length: Optional[int] = 5000

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward=512, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = SinusoidalPositionalEncoding(emb_size, dropout=dropout, maxlen=max_seq_len)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
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
    return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

def translate_sentence(src_sentence: str, max_len: int = 50):
    global model, vocab_transform, text_transform

    model.eval()
    with torch.no_grad():
        src_tensor = text_transform['en'](src_sentence).unsqueeze(0).to(DEVICE)
        src_padding_mask = (src_tensor == PAD_IDX).to(DEVICE)
        memory = model.encode(src_tensor, src_mask=None, src_padding_mask=src_padding_mask)
        memory = memory.to(DEVICE)
        ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)

        for _ in range(max_len - 1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_idx)], dim=1)
            if next_word_idx == EOS_IDX:
                break

        tgt_tokens = [vocab_transform['te'].get_itos()[i] for i in ys.squeeze(0).tolist()]
        return " ".join([token for token in tgt_tokens if token not in special_symbols])

@translation_router.post("/translate")
async def translate(request: TranslationRequest):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        english_text = request.sentence.strip()
        if not english_text:
            raise HTTPException(status_code=400, detail="Empty text provided")

        max_length = request.max_length
        telugu_translation = translate_sentence(english_text, max_len=max_length)

        return {
            'success': True,
            'input': english_text,
            'translation': telugu_translation,
            'language_pair': 'en-te'
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

def initialize_model():
    global model, vocab_transform, token_transform, text_transform
    try:
        vocab_transform = {
            'en': torch.load( os.path.join(BASE_DIR,'vocab_transform_en.pt'), map_location=DEVICE),
            'te': torch.load( os.path.join(BASE_DIR,'vocab_transform_te.pt'), map_location=DEVICE)
        }
        SRC_VOCAB_SIZE = len(vocab_transform['en'])
        TGT_VOCAB_SIZE = len(vocab_transform['te'])

        token_transform = {
            'en': get_tokenizer('spacy', language='en_core_web_sm'),
            'te': tokenize_te_placeholder
        }

        text_transform = {
            ln: sequential_transforms(token_transform[ln], vocab_transform[ln], tensor_transform)
            for ln in ['en', 'te']
        }

        model_instance = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
                                            SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT)
        model_instance.to(DEVICE)
        model_instance.load_state_dict(torch.load( os.path.join(BASE_DIR,'transformer_eng_tel_scratch_full_data.pt') , map_location=DEVICE))
        model_instance.eval()
        model = model_instance
        print("Translation Model loaded successfully!")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to initialize translation model: {str(e)}")

# Initialize model when module is imported
initialize_model()