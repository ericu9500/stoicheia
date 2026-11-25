"""Region/century metadata conditioning: vocab parsing, collate propagation, model gating."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insc" / "data"))
from train.collate import pack_batch, collate, UNK_REGION, UNK_CENTURY
from train.noising import NoiseConfig
from model.char_bert import CharBertConfig, CharBertEncoder
from meta_vocab import (parse_year, region_to_id, record_century_id, year_to_century_id,
                         N_REGION, N_CENTURY, CENTURY_LO)


def test_collate_unk_constants_match_meta_vocab():
    """train/collate.py hardcodes UNK_REGION/UNK_CENTURY (to avoid a base-pretraining ->
    insc-only-package import dependency) instead of importing meta_vocab's. Guard against
    them drifting apart if meta_vocab's region/century bucketing is ever changed."""
    assert UNK_REGION == region_to_id(None)
    assert UNK_CENTURY == record_century_id(None, None)


def test_data_py_unk_constants_match_meta_vocab():
    """train/data.py's ShardSet also hardcodes its own UNK_REGION/UNK_CENTURY copy, same
    reason as collate.py's. Same drift guard."""
    from train.data import UNK_REGION as DATA_UNK_REGION, UNK_CENTURY as DATA_UNK_CENTURY
    assert DATA_UNK_REGION == region_to_id(None)
    assert DATA_UNK_CENTURY == record_century_id(None, None)


def test_parse_year_rejects_sentinels_and_junk():
    assert parse_year("-400") == -400
    assert parse_year("1") == 1
    assert parse_year(None) is None
    assert parse_year("") is None
    assert parse_year("-") is None
    assert parse_year("999") is None
    assert parse_year("-999") is None
    assert parse_year("NULL") is None
    assert parse_year("null34") is None
    assert parse_year("-0") is None


def test_region_to_id_known_and_unknown():
    assert region_to_id("Attica") != UNK_REGION
    assert region_to_id("Nowhereland") == UNK_REGION
    assert region_to_id(None) == UNK_REGION


def test_record_century_id_midpoint_and_fallback():
    c1 = record_century_id("-400", "-301")     # midpoint -350 -> same bucket as -350 alone
    c2 = year_to_century_id(-350)
    assert c1 == c2
    assert record_century_id(None, "-301") == year_to_century_id(-301)
    assert record_century_id(None, None) == 15
    assert record_century_id("999", "-999") == UNK_CENTURY  # both sentinel -> UNK


def _fake_rec(L, region_id=None, century_id=None, seed=0):
    rng = np.random.default_rng(seed)
    chars = rng.integers(0, 24, L).astype(np.uint8)
    boundary = np.zeros(L, np.uint8); boundary[-1] = 2
    d = dict(chars=chars, boundary=boundary, dia=np.zeros(L, np.uint8),
              cap=np.zeros(L, np.uint8), punct=np.zeros(L, np.uint8))
    if region_id is not None:
        d["region_id"] = region_id
    if century_id is not None:
        d["century_id"] = century_id
    return d


def test_pack_batch_propagates_region_century_and_defaults_unk():
    cfg = NoiseConfig(w_span=0.5, w_word=0.5, w_elastic=0.0, w_iid=0.0, w_halfword=0.0,
                      w_substitute=0.0)
    g = torch.Generator().manual_seed(0)
    records = [_fake_rec(64, region_id=3, century_id=7), _fake_rec(64)]  # 2nd: no metadata
    it = iter(records)
    batch = pack_batch(it, cfg, 128, 1, g)
    assert "region" in batch and "century" in batch
    seg = batch["seg_id"][0]
    reg = batch["region"][0]
    cen = batch["century"][0]
    doc1 = seg == 1
    doc2 = seg == 2
    assert (reg[doc1] == 3).all() and (cen[doc1] == 7).all()
    assert (reg[doc2] == UNK_REGION).all() and (cen[doc2] == UNK_CENTURY).all()


def test_collate_propagates_region_century():
    cfg = NoiseConfig(w_span=0.5, w_word=0.5, w_elastic=0.0, w_iid=0.0, w_halfword=0.0,
                      w_substitute=0.0)
    g = torch.Generator().manual_seed(0)
    records = [_fake_rec(64, region_id=5, century_id=2)]
    batch = collate(records, cfg, 128, g)
    seqlen = (batch["seg_id"][0] > 0).sum().item()
    assert (batch["region"][0, :seqlen] == 5).all()
    assert (batch["century"][0, :seqlen] == 2).all()


def test_model_ignores_region_century_when_disabled():
    """Even though every batch now always carries region/century keys, a model built with
    n_region=n_century=0 (every existing checkpoint) must produce identical output whether
    or not those keys vary -- it never looks at them."""
    torch.manual_seed(0)
    cfg = CharBertConfig(attn_impl="sdpa", d_model=32, n_heads=4, depth=1, char_window=0)
    m = CharBertEncoder(cfg)
    assert m.e_region is None and m.e_century is None
    B, T = 2, 16
    base = dict(input_ids=torch.randint(0, 24, (B, T)), seg_id=torch.ones(B, T, dtype=torch.long),
                boundary=torch.zeros(B, T, dtype=torch.long), dia=torch.zeros(B, T, dtype=torch.long),
                punct=torch.zeros(B, T, dtype=torch.long))
    b1 = dict(base, region=torch.zeros(B, T, dtype=torch.long), century=torch.zeros(B, T, dtype=torch.long))
    b2 = dict(base, region=torch.randint(0, N_REGION, (B, T)), century=torch.randint(0, N_CENTURY, (B, T)))
    with torch.no_grad():
        o1 = m(b1)["char"]
        o2 = m(b2)["char"]
    assert torch.equal(o1, o2)
