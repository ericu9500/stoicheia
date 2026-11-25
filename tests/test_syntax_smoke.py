"""Smoke test for the joint tagger+parser pipeline's tensor plumbing: a tiny, randomly
initialized encoder + JointModel, fed a synthetic mini-batch, produces correctly-shaped
tag/lemma/UPOS/arc/label predictions end to end. Does NOT test tagging/parsing quality
(real weights, real treebank) -- only that packing/pooling/scalar-mix/biaffine survive a
refactor without shape or index errors."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.char_bert import CharBertConfig
from tagger.backbone import CharBertWithHidden
from tagger.edits import LabelVocab
from tagger.model import TaggerConfig
from parser.biaffine import ParserConfig
from parser.joint_model import JointModel


def _tiny_vocab():
    return LabelVocab(
        scripts=[("", "")],
        xpos_alpha=[["-", "n", "v"] for _ in range(9)],
        tags=["n-s---fa-", "v2spma---"],
        upos=["NOUN", "VERB"],
        lex_ft={}, lex_f={}, nongreek={},
    )


def test_joint_model_forward_pass():
    enc_cfg = CharBertConfig(d_model=32, n_heads=4, depth=2, char_window=8, attn_impl="sdpa")
    encoder = CharBertWithHidden(enc_cfg)
    encoder.return_layers = True
    encoder.eval()

    vocab = _tiny_vocab()
    tcfg = TaggerConfig(pool="mean", use_cap=False, scalar_mix=True, w_flat=0.0)
    pcfg = ParserConfig(d_arc=16, d_rel=8, dropout=0.0, n_labels=4)
    W = 6  # max words per row

    model = JointModel(encoder, vocab, tcfg, pcfg, W=W)
    model.eval()

    # synthetic mini-batch: 2 short "sentences" of 3 words each, packed into one row
    T = 20
    B = 1
    input_ids = torch.randint(0, 24, (B, T))
    boundary = torch.zeros(B, T, dtype=torch.long)
    boundary[:, [4, 9, 14, 19]] = 1  # word ends
    boundary[:, 19] = 2              # sentence end
    dia = torch.zeros(B, T, dtype=torch.long)
    punct = torch.zeros(B, T, dtype=torch.long)
    seg_id = torch.zeros(B, T, dtype=torch.long)
    # word_id: -1 for non-final char positions is not required by pool_words (mean over all
    # positions sharing a word id); assign each 5-char span to one word slot 0..3
    word_id = torch.tensor([[w for w in range(4) for _ in range(5)]], dtype=torch.long)

    # one sentence (id 0) occupying word slots 0..3 of this single packed row
    slots = [[(0, 0), (0, 1), (0, 2), (0, 3)]]

    batch = dict(input_ids=input_ids, boundary=boundary, dia=dia, punct=punct,
                 seg_id=seg_id, word_id=word_id, slots=slots)

    with torch.no_grad():
        tag_out, arc_scores, rel_scores, word_mask, sent_ids = model(batch)

    # tagger heads: factored XPOS (list of 9 per-position logit tensors), UPOS, lemma-script
    assert len(tag_out["xpos"]) == 9
    assert tag_out["xpos"][0].shape[:2] == (B, W)
    assert tag_out["upos"].shape[:2] == (B, W)
    assert tag_out["script"].shape[:2] == (B, W)
    # biaffine arc/label scores: one sentence, 4 real words + 1 root column
    assert arc_scores.shape[0] == 1
    assert arc_scores.shape[1] == word_mask.shape[1]
    assert rel_scores.shape[0] == 1
    assert sent_ids == [0]
