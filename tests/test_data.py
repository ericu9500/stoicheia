import os
from pathlib import Path

import numpy as np
import pytest

from tagger.backbone import restore_polytonic
from tagger.conllu import read_conllu, write_conllu
from tagger.dataset import TaggerDataset, batch_rows, encode_sentence, encode_word, pack_rows
from tagger.edits import LabelVocab, form_key

KFOLD = Path(os.path.expandvars(os.environ.get("TAGGER_KFOLD",
             "$STOICHEIA_DATA/treebanks/oga_repo/kfold")))


pytestmark = pytest.mark.skipif(
    not (KFOLD / "dev0.conllu").is_file(),
    reason="TAGGER_KFOLD not set to a checkout of the OGA treebank k-fold split")


@pytest.fixture(scope="module")
def sents():
    out = []
    for i, s in enumerate(read_conllu(KFOLD / "dev0.conllu")):
        if i >= 200:
            break
        out.append(s)
    return out


@pytest.fixture(scope="module")
def vocab(sents):
    return LabelVocab.build(sents, lambda f: encode_word(f) is not None)


def test_conllu_roundtrip(tmp_path, sents):
    preds = [[(t.lemma, t.upos, t.xpos) for t in s.tokens] for s in sents]
    write_conllu(sents, preds, tmp_path / "rt.conllu")
    back = list(read_conllu(tmp_path / "rt.conllu"))
    assert len(back) == len(sents)
    for a, b in zip(sents, back):
        assert [t.form for t in a.tokens] == [t.form for t in b.tokens]
        assert [t.xpos for t in a.tokens] == [t.xpos for t in b.tokens]


def test_span_alignment(sents, vocab):
    """restore_polytonic over each word span must reproduce the FORM (mod key-folding)."""
    checked = 0
    for s in sents:
        e = encode_sentence(s, vocab)
        if e is None:
            continue
        # per-word cap plane is discarded in SentEnc; re-encode per word for the check
        for t, span in zip(s.tokens, e.spans):
            if span is None:
                continue
            enc = encode_word(t.form)
            chars, dia, cap = enc
            s0, s1 = span
            assert np.array_equal(e.chars[s0:s1], chars)
            words = restore_polytonic(chars, dia, cap, np.array([0] * (len(chars) - 1) + [1]))
            assert len(words) == 1
            # restored span == FORM restricted to Greek letters+marks (the char stream
            # drops apostrophes/brackets/digits; macron/breve/underdot are stripped marks)
            import unicodedata
            target = "".join(
                c for c in unicodedata.normalize("NFD", t.form)
                if unicodedata.category(c) in ("Ll", "Lu", "Lo") or c in "́̀͂̓̔̈ͅ")
            target = unicodedata.normalize("NFC", target)
            assert form_key(words[0]).replace("ς", "σ") == form_key(target).replace("ς", "σ"), (t.form, words[0])
            checked += 1
    assert checked > 500


def test_packing_invariants(sents, vocab):
    ds = TaggerDataset(sents, vocab, T=1024, W=256)
    rows, trunc = pack_rows(ds.encs, 1024, 256)
    batch = batch_rows(rows[:4], 1024, 256)
    ids, seg, wid = batch["input_ids"], batch["seg_id"], batch["word_id"]
    assert ids.shape == seg.shape == wid.shape
    # pads are exactly where seg==0, and pads carry pad_id
    assert bool(((ids == 26) == (seg == 0)).all())
    # every labeled word slot has at least one char pointing at it
    for b in range(ids.shape[0]):
        labeled = (batch["y_script"][b] != -100).nonzero().flatten().tolist()
        pointed = set(wid[b][wid[b] >= 0].tolist())
        assert set(labeled) <= pointed
    # slots align with labels
    for b, rs in enumerate(batch["slots"]):
        assert len(rs) == len(set(wid[b][wid[b] >= 0].tolist()))
