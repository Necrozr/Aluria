"""Tokenizador baseado em chunks (nao BPE, nao WordPiece, nao LLM).

Ideia: divide o texto em blocos fixos de caracteres (chunks) e trata
cada chunk distinto como um simbolo. Sem merges, sem atencao, sem vocab
pre-treinado gigante. E propositalmente cru: serve ao experimento de
linguagem emergente, onde a estrutura deve surgir do uso, nao do tokenizer.
"""
import json
from collections import Counter


PAD = "<pad>"
UNK = "<unk>"


class ChunkTokenizer:
    def __init__(self, chunk_size=4, vocab_size=2000, stride=None, lowercase=False):
        self.chunk_size = int(chunk_size)
        self.vocab_size = int(vocab_size)
        self.stride = int(stride) if stride else int(chunk_size)
        self.lowercase = bool(lowercase)
        # id 0 = PAD, id 1 = UNK
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}

    def _norm(self, text):
        if self.lowercase:
            return text.lower()
        return text

    def _chunk_iter(self, text):
        text = self._norm(text)
        k, s = self.chunk_size, self.stride
        for i in range(0, len(text), s):
            c = text[i:i + k]
            if not c:
                break
            # ultimo pedaco menor que k: completa com espaco para manter
            # decodificacao por concatenacao simples
            if len(c) < k:
                c = c + " " * (k - len(c))
            yield c

    def fit(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(self._chunk_iter(t))
        # reserva 2 ids especiais
        keep = max(0, self.vocab_size - 2)
        most = [c for c, _ in cnt.most_common(keep)]
        self.itos = [PAD, UNK] + most
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        return self

    def encode(self, text):
        return [self.stoi.get(c, 1) for c in self._chunk_iter(text)]

    def decode(self, ids):
        chars = "".join(self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids)
        return chars.strip()

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "chunk_size": self.chunk_size,
                "vocab_size": self.vocab_size,
                "stride": self.stride,
                "lowercase": self.lowercase,
                "itos": self.itos,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(chunk_size=d["chunk_size"], vocab_size=d.get("vocab_size", 2000),
                  stride=d.get("stride"), lowercase=d.get("lowercase", False))
        tok.itos = d["itos"]
        tok.stoi = {c: i for i, c in enumerate(tok.itos)}
        return tok
