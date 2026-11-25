"""eval_span_whole: the realistic restoration task ('fill in a blank in an inscription/papyrus
AS EDITED, real lacunae intact elsewhere in the same document') must never place its scored
synthetic gap on top of a real lacuna, and must run cleanly end to end."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insc" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insc" / "data"))

from model.char_bert import CharBertConfig, CharBertEncoder
import restore


def _tiny_model():
    torch.manual_seed(0)
    cfg = CharBertConfig(attn_impl="sdpa", d_model=32, n_heads=4, depth=1, char_window=0)
    m = CharBertEncoder(cfg)
    m.eval()
    return m


def _fake_records_with_lacuna(n, L, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        chars = rng.integers(0, 24, L).astype(np.int64)
        boundary = np.zeros(L, np.int64); boundary[4::5] = 1; boundary[-1] = 2
        is_real_lacuna = np.zeros(L, dtype=bool)
        lac_s = int(rng.integers(10, L - 15))
        lac_len = 5
        chars[lac_s:lac_s + lac_len] = 24
        boundary[lac_s:lac_s + lac_len] = 3
        is_real_lacuna[lac_s:lac_s + lac_len] = True
        recs.append(dict(chars=chars, boundary=boundary, is_real_lacuna=is_real_lacuna,
                          region_id=3, century_id=5, _lac=(lac_s, lac_s + lac_len)))
    return recs


def test_eval_span_whole_runs_end_to_end():
    model = _tiny_model()
    recs = _fake_records_with_lacuna(5, 100)
    r = restore.eval_span_whole(model, recs, L=3, device=torch.device("cpu"), n=5, beam_width=4)
    assert r["n"] > 0
    assert 0 <= r["top1"] <= 1 and 0 <= r["top20"] <= 1


def test_eval_span_whole_never_overlaps_real_lacuna():
    model = _tiny_model()
    recs = _fake_records_with_lacuna(30, 60, seed=1)
    seen_gaps = []
    orig = restore.beam_restore

    def spy(model, chars, gap, bnd_row, device, *args, **kwargs):
        seen_gaps.append(list(gap))
        return orig(model, chars, gap, bnd_row, device, *args, **kwargs)

    restore.beam_restore = spy
    try:
        r = restore.eval_span_whole(model, recs, L=2, device=torch.device("cpu"), n=30,
                                    beam_width=2)
    finally:
        restore.beam_restore = orig
    assert r["n"] > 0
    assert len(seen_gaps) == r["n"]
    for gap, rec in zip(seen_gaps, recs):
        lac_s, lac_e = rec["_lac"]
        assert not any(lac_s <= p < lac_e for p in gap), \
            f"synthetic gap {gap} overlaps real lacuna [{lac_s},{lac_e})"


def test_eval_span_whole_skips_records_with_no_valid_position():
    """A record entirely consumed by real lacunae (or too short) has no valid place for the
    synthetic gap -- must be skipped, not crash."""
    model = _tiny_model()
    L = 20
    chars = np.zeros(L, np.int64) + 24  # all MASK
    boundary = np.full(L, 3, np.int64)
    is_real_lacuna = np.ones(L, dtype=bool)
    recs = [dict(chars=chars, boundary=boundary, is_real_lacuna=is_real_lacuna,
                 region_id=3, century_id=5)]
    r = restore.eval_span_whole(model, recs, L=3, device=torch.device("cpu"), n=5, beam_width=2)
    assert r["n"] == 0
