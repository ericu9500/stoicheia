"""Smoke test for the restoration pipeline's tensor plumbing: a tiny, randomly
initialized CharBertEncoder, fed a masked (gap) span, produces correctly-shaped
predictions end to end. This does NOT test restoration quality (real weights,
real corpora) -- only that packing/masking/forward/argmax-decode survive a refactor
without shape or index errors."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.char_bert import CharBertConfig, CharBertEncoder
from data.normalize import ALPHABET, Stats, normalize_record

MASK, UNK_BND, UNK_DIA, UNK_PUNCT = 24, 3, 48, 6


def _tiny_encoder():
    cfg = CharBertConfig(d_model=32, n_heads=4, depth=2, char_window=8, attn_impl="sdpa")
    enc = CharBertEncoder(cfg)
    enc.eval()
    return enc


def test_restoration_forward_pass_with_masked_gap():
    text = "ενἀρχῇἦνὁλόγοςκαὶὁλόγοςἦνπρὸςτὸνθεόν"
    st = Stats()
    nr = normalize_record(text, st, with_punct=True)
    assert nr is not None
    chars, boundary, dia, cap, punct = nr
    n = len(chars)
    assert n > 10

    # mask a small span (simulating a lacuna of known length) in letters + boundary
    gap = slice(3, 7)
    chars = chars.copy().astype(np.int64)
    boundary = boundary.copy().astype(np.int64)
    chars[gap] = MASK
    boundary[gap] = UNK_BND

    batch = dict(
        input_ids=torch.from_numpy(chars)[None],
        boundary=torch.from_numpy(boundary)[None],
        dia=torch.from_numpy(dia.astype(np.int64))[None],
        punct=torch.from_numpy(punct.astype(np.int64))[None],
        seg_id=torch.zeros(1, n, dtype=torch.long),
    )

    enc = _tiny_encoder()
    with torch.no_grad():
        out = enc(batch)

    for key in ("char", "boundary", "dia", "cap", "punct"):
        assert key in out
        assert out[key].shape[:2] == (1, n)

    pred_chars = out["char"].argmax(-1)[0]
    assert pred_chars.shape == (n,)
    # masked positions should be assignable a predicted letter id in range
    assert int(pred_chars[gap.start]) in range(27)
