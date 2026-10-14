"""CTC output units: a subset of Gemma tokenizer pieces chosen from Norwegian training text.

Keeping Gemma pieces (rather than a new SentencePiece model) lets the output head start from
EmbeddingGemma's text embeddings for the same pieces, as QUAN-Transcribe does (`head_init: text`).
Words containing a piece outside the subset fall back to single-character pieces, which are
always included, so every clean text can be encoded.
"""

import collections
import json
from pathlib import Path

from transformers import AutoTokenizer

from .text import LETTERS

BLANK = 0
SPACE = "▁"


class Units:
    def __init__(self, pieces: list[str], gemma_ids: list[int], model_id: str):
        self.pieces = pieces  # index 0 is the CTC blank
        self.gemma_ids = gemma_ids
        self.model_id = model_id
        self.index = {p: i for i, p in enumerate(pieces)}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def __len__(self) -> int:
        return len(self.pieces)

    @staticmethod
    def char_pieces() -> list[str]:
        return sorted(LETTERS) + [SPACE + c for c in sorted(LETTERS)]

    def _word_pieces(self, word: str) -> list[str]:
        pieces = self.tokenizer.tokenize(" " + word)
        if all(p in self.index for p in pieces):
            return pieces
        return [SPACE + word[0], *word[1:]]

    def encode(self, text: str) -> list[int]:
        return [self.index[p] for w in text.split() for p in self._word_pieces(w)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids if i != BLANK).replace(SPACE, " ").strip()

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"model_id": self.model_id, "pieces": self.pieces, "gemma_ids": self.gemma_ids}, ensure_ascii=False)
        )

    @classmethod
    def load(cls, path: Path) -> "Units":
        d = json.loads(Path(path).read_text())
        return cls(d["pieces"], d["gemma_ids"], d["model_id"])

    @classmethod
    def build(cls, texts, model_id: str, min_count: int = 5) -> "Units":
        tok = AutoTokenizer.from_pretrained(model_id)
        counts = collections.Counter()
        for text in texts:
            counts.update(tok.tokenize(" " + text))
        chars = cls.char_pieces()
        kept = sorted({p for p, n in counts.items() if n >= min_count} | set(chars))
        vocab = tok.get_vocab()
        kept = [p for p in kept if p in vocab]
        missing = set(chars) - set(kept)
        if missing:
            raise ValueError(f"Gemma vocabulary lacks character pieces: {sorted(missing)}")
        pieces = ["<blank>", *kept]
        gemma_ids = [-1, *(vocab[p] for p in kept)]
        return cls(pieces, gemma_ids, model_id)
