"""Pack OGA sentences into char-arm model batches, token-aligned.

char arm: tagger.dataset.encode_sentence (raw FORM encoding) -> Stoicheia (word-pooled).
Skips exactly the tokens tagger.dataset.encode_word() rejects (no Greek letters).

NOTE: this release drops the lemma-arm packing helpers (LemmaVocabMap, tag_sentences,
pack_lemma_rows, batch_lemma) that fed a LemmaDiff-grc encoder — LemmaDiff-grc is a separate,
unpublished side-repo. See parser/model.py's module docstring for the rationale.
"""
from __future__ import annotations

import numpy as np
import torch

from tagger.dataset import encode_sentence, batch_rows as char_batch_rows, pack_rows as char_pack_rows


def n_encodable(sent):
    return sum(1 for t in sent.tokens if encode_sentence_word_ok(t.form))


def encode_sentence_word_ok(form):
    from tagger.dataset import encode_word
    return encode_word(form) is not None


# ---------------------------------------------------------------- char arm (reuse tagger's)

def pack_char_rows(sents, T=2048, W=384):
    encs = [encode_sentence(s) for s in sents]
    pairs = [(i, e) for i, e in enumerate(encs) if e is not None]
    rows, trunc = char_pack_rows([e for _, e in pairs], T, W)
    return rows, pairs, trunc


def batch_char(rows, T, W, device):
    b = char_batch_rows(rows, T, W)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}, b["slots"]

# lemma-arm packing helpers (LemmaVocabMap, tag_sentences, pack_lemma_rows, batch_lemma) removed
# for this release — see module docstring above.
