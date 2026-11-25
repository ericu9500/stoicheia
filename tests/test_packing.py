"""Sequence-packing correctness: docs don't cross, fill ratio high, labels only at masked."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.collate import pack_batch
from train.noising import NoiseConfig


def _fake_records(n, lens, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        L = lens[i % len(lens)]
        chars = rng.integers(0, 24, L).astype(np.uint8)
        boundary = np.zeros(L, np.uint8)
        boundary[4::5] = 1
        boundary[-1] = 2
        yield dict(chars=chars, boundary=boundary,
                   dia=np.zeros(L, np.uint8), cap=np.zeros(L, np.uint8),
                   punct=np.zeros(L, np.uint8))


def test_packing_no_doc_crossing_and_fill():
    cfg = NoiseConfig(w_span=0.5, w_word=0.5, w_elastic=0.0, w_iid=0.0, w_halfword=0.0,
                      w_substitute=0.0)  # length-preserving
    g = torch.Generator().manual_seed(0)
    T, rows = 512, 4
    it = _fake_records(10000, [120, 200, 336, 90], seed=1)
    batch = pack_batch(it, cfg, T, rows, g)
    seg = batch["seg_id"]
    assert seg.max().item() >= 2, "no packing occurred"
    for b in range(rows):
        s = seg[b][seg[b] > 0].tolist()
        runs = []
        for v in s:
            if not runs or runs[-1][0] != v:
                runs.append([v, 0])
            runs[-1][1] += 1
        seen = [r[0] for r in runs]
        assert len(seen) == len(set(seen)), f"doc ids interleave in row {b}: {seen}"
    fill = (seg > 0).float().mean().item()
    assert fill > 0.70, f"fill only {fill:.2f}"


def test_packing_labels_only_at_masked():
    cfg = NoiseConfig(w_span=1.0, w_word=0.0, w_elastic=0.0, w_iid=0.0, w_halfword=0.0,
                      w_substitute=0.0)
    g = torch.Generator().manual_seed(2)
    batch = pack_batch(_fake_records(10000, [200], 3), cfg, 512, 3, g)
    masked = batch["input_ids"] == 24
    supervised = batch["labels"] != -100
    assert torch.equal(masked, supervised)


def _fake_records_with_lacuna(n, L, lac_start, lac_len, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        chars = rng.integers(0, 24, L).astype(np.uint8)
        chars[lac_start:lac_start + lac_len] = 24  # MASK_ID, matching text_to_full_planes
        boundary = np.zeros(L, np.uint8)
        boundary[4::5] = 1
        boundary[lac_start:lac_start + lac_len] = 3  # UNK_BND
        boundary[-1] = 2
        is_real_lacuna = np.zeros(L, dtype=bool)
        is_real_lacuna[lac_start:lac_start + lac_len] = True
        yield dict(chars=chars, boundary=boundary,
                   dia=np.zeros(L, np.uint8), cap=np.zeros(L, np.uint8),
                   punct=np.zeros(L, np.uint8), is_real_lacuna=is_real_lacuna,
                   region_id=5, century_id=7)


def test_pack_batch_real_lacuna_never_supervised():
    """A record's real-lacuna span (is_real_lacuna=True) must never appear as a char label
    or an aux label, no matter which (fixed-length) synthetic pattern gets drawn. Uses a
    single record that exactly fills one row (no packing/truncation) so post-pack positions
    map 1:1 to the original record, letting the lacuna span be checked directly at a fixed
    index -- excludes the elastic pattern (sequence-length-changing, so a fixed index no
    longer maps to the same position; already covered separately in test_noising.py)."""
    cfg = NoiseConfig(w_elastic=0.0, w_span=0.3, w_word=0.3, w_iid=0.2, w_halfword=0.1,
                      w_substitute=0.1)
    lac_start, lac_len = 50, 10
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        recs = _fake_records_with_lacuna(1, 200, lac_start, lac_len, seed=seed)
        batch = pack_batch(recs, cfg, 200, 1, g)
        lab = batch["labels"][0, lac_start:lac_start + lac_len]
        assert (lab == -100).all(), f"seed={seed}"
        assert (batch["bnd_lab"][0, lac_start:lac_start + lac_len] == -100).all()
        assert (batch["dia_lab"][0, lac_start:lac_start + lac_len] == -100).all()
        assert (batch["cap_lab"][0, lac_start:lac_start + lac_len] == -100).all()
        assert (batch["punct_lab"][0, lac_start:lac_start + lac_len] == -100).all()


def test_pack_batch_metadata_dropout_forces_unk_sometimes():
    cfg = NoiseConfig(p_region_none=1.0, p_century_none=0.0)
    g = torch.Generator().manual_seed(0)
    batch = pack_batch(_fake_records_with_lacuna(2000, 100, 20, 5, seed=2), cfg, 256, 2, g)
    present = batch["region"][batch["seg_id"] > 0]
    assert (present == 14).all(), "p_region_none=1.0 must force UNK_REGION everywhere"
    present_c = batch["century"][batch["seg_id"] > 0]
    assert (present_c == 7).all(), "p_century_none=0.0 must never drop the true century"


def test_pack_batch_metadata_dropout_default_off_matches_prior_behavior():
    cfg = NoiseConfig()  # p_region_none/p_century_none default to 0.0
    g = torch.Generator().manual_seed(0)
    batch = pack_batch(_fake_records_with_lacuna(2000, 100, 20, 5, seed=3), cfg, 256, 2, g)
    present = batch["region"][batch["seg_id"] > 0]
    assert (present == 5).all(), "dropout defaults to 0.0 -- true region always kept"
