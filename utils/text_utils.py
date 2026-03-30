import re

def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9 ]", "", text)

def encode_question(q, vocab, max_len=20):
    tokens = q.split()
    enc = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
    enc = enc[:max_len] + [vocab["<PAD>"]] * (max_len - len(enc))
    return enc
