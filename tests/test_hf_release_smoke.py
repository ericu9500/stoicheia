"""Smoke test for the HuggingFace-Hub release wrappers (hf_release/): a tiny,
randomly initialized CharBertModel + CharBertProcessor round-trips text through
encode -> forward -> decode. This does NOT test model quality (random weights) --
only that the Hub-facing config/model/processor plumbing survives a refactor, and
in particular that decode_restoration/decode_diacritics/decode_boundaries actually
reinsert the punct plane (comma/middle-dot/colon/period/semicolon)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hf_release.configuration_char_bert import CharBertConfig
from hf_release.modeling_char_bert import CharBertModel
from hf_release.processing_char_bert import CharBertProcessor, MASK, UNK_BND, UNK_DIA, UNK_PUNCT


def _onehot_logits(indices, n_classes):
    """[T] class indices -> [1, T, n_classes] logits that argmax back to `indices`."""
    t = len(indices)
    logits = torch.zeros(1, t, n_classes)
    for i, c in enumerate(indices):
        logits[0, i, c] = 10.0
    return logits


def test_processor_round_trips_punctuation_with_no_masking():
    proc = CharBertProcessor()
    text = "λόγος, ἔργον· καλόν: τέλος. τί;"
    batch = proc(text)
    cap = batch["_cap"]
    out = proc._restore_polytonic(
        batch["input_ids"][0].tolist(), batch["dia"][0].tolist(), cap,
        batch["boundary"][0].tolist(), batch["punct"][0].tolist(),
    )
    assert out == text


def test_decode_restoration_fills_masked_gap_and_keeps_punctuation():
    proc = CharBertProcessor()
    text = "λόγος, ἔργον καλόν."
    batch = proc(text)

    # mask a gap inside "ἔργον" only -- well clear of the comma's punct-plane index,
    # so this isolates "does restoration correctly fill the gap" from "does
    # untouched punctuation survive decode" as two independent assertions.
    gap = slice(7, 10)
    orig_chars = batch["input_ids"][0].tolist()
    orig_dia = batch["dia"][0].tolist()
    orig_bnd = batch["boundary"][0].tolist()
    batch["input_ids"][0, gap] = MASK
    batch["boundary"][0, gap] = UNK_BND
    batch["dia"][0, gap] = UNK_DIA
    batch["punct"][0, gap] = UNK_PUNCT

    n = batch["input_ids"].shape[1]
    fake_out = SimpleNamespace(
        char=_onehot_logits(orig_chars, 27),
        boundary=_onehot_logits(orig_bnd, 3),
        dia=_onehot_logits(orig_dia, 48),
        cap=_onehot_logits([0] * n, 2),
        punct=_onehot_logits([0] * n, 6),
    )
    decoded = proc.decode_restoration(fake_out, batch)
    assert decoded == text  # perfect predictions -> exact reconstruction, comma included


def test_decode_diacritics_keeps_punctuation_and_boundary():
    proc = CharBertProcessor()
    text = "λόγος, ἔργον καλόν."
    batch = proc(text, mask_planes=["dia"])
    n = batch["input_ids"].shape[1]
    fake_out = SimpleNamespace(dia=_onehot_logits([0] * n, 48))
    decoded = proc.decode_diacritics(fake_out, batch)
    assert "," in decoded


def test_decode_boundaries_keeps_punctuation():
    proc = CharBertProcessor()
    text = "λόγος, ἔργον καλόν."
    batch = proc(text)
    orig_bnd = batch["boundary"][0].tolist()
    # predict the boundary plane perfectly (as a real trained model would for
    # already-spaced input) so the punct-plane flush points line up correctly
    fake_out = SimpleNamespace(boundary=_onehot_logits(orig_bnd, 3))
    decoded = proc.decode_boundaries(fake_out, batch)
    assert decoded == text


def test_decode_restoration_handles_fully_masked_dia_and_boundary():
    """Joint restoration: mask_planes=["dia", "boundary"] forces those planes to
    UNK everywhere, not just inside the '-' gap. decode_restoration must fill dia/
    boundary wherever THAT plane is UNK (not only where chars==MASK), or it crashes
    trying to unpack the raw UNK_DIA sentinel as if it were a real diacritic state."""
    proc = CharBertProcessor()
    text = "λογος--εργον"
    batch = proc(text, mask_planes=["dia", "boundary"], has_boundaries=False)
    n = batch["input_ids"].shape[1]

    # every position's dia/boundary is UNK; only the '-' gap's chars are MASK
    assert (batch["dia"][0] == UNK_DIA).all()
    assert (batch["boundary"][0] == UNK_BND).all()

    fake_out = SimpleNamespace(
        char=_onehot_logits(batch["input_ids"][0].tolist(), 27),
        boundary=_onehot_logits([0] * n, 3),
        dia=_onehot_logits([0] * n, 48),
        cap=_onehot_logits([0] * n, 2),
        punct=_onehot_logits([0] * n, 6),
    )
    decoded = proc.decode_restoration(fake_out, batch)  # must not raise KeyError
    assert isinstance(decoded, str) and decoded


def test_restore_elastic_sweeps_widths_and_ranks_by_confidence():
    cfg = CharBertConfig(d_model=32, n_heads=4, depth=2, char_window=8, attn_impl="sdpa")
    model = CharBertModel(cfg)
    model.eval()
    proc = CharBertProcessor()

    text = "λογ[3±2]εργον"  # candidate widths 1..5
    best_text, best_width, candidates = proc.restore_elastic(model, text)

    widths = [w for w, _, _ in candidates]
    assert set(widths) == set(range(1, 6))  # candidate widths 3-2..3+2, clamped to >= 1
    assert best_width in range(1, 6)
    assert isinstance(best_text, str) and best_text
    # sorted best-first (highest/least-negative logp first)
    logps = [c[2] for c in candidates]
    assert logps == sorted(logps, reverse=True)


def test_tiny_model_forward_pass_shapes():
    """Loose end-to-end check that CharBertModel + CharBertOutput plug into the
    processor without shape/attribute errors (separate from decode correctness,
    which is covered above with deterministic fake logits)."""
    cfg = CharBertConfig(d_model=32, n_heads=4, depth=2, char_window=8, attn_impl="sdpa")
    model = CharBertModel(cfg)
    model.eval()
    proc = CharBertProcessor()
    batch = proc("λόγος καλόν")
    n = batch["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**{k: v for k, v in batch.items() if k != "_cap"})
    for key in ("char", "boundary", "dia", "cap", "punct"):
        assert getattr(out, key).shape[:2] == (1, n)
    # exercise the full decode path end-to-end (quality not asserted -- random weights)
    assert isinstance(proc.decode_restoration(out, batch), str)
