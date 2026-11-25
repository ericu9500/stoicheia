"""Shard-level region_id/century_id plumbing: written by insc build_shards.py, read by
ShardSet/MultiTierLoader, must survive _window()'s slicing (scalar, not per-char)."""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.data import DataConfig, MultiTierLoader, ShardSet, TierSpec, UNK_CENTURY, UNK_REGION

PLANES = ("chars", "boundary", "dia", "cap", "punct")


def _write_shard(d, n_records=6, rec_len=300, with_meta=True):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bufs = {p: [] for p in PLANES}
    rows = {"offset": [], "length": [], "tier": [], "clean": [], "dup_frac": []}
    if with_meta:
        rows["region_id"] = []
        rows["century_id"] = []
    off = 0
    for i in range(n_records):
        for p in PLANES:
            bufs[p].append(rng.integers(0, 4, rec_len).astype(np.uint8))
        rows["offset"].append(off); rows["length"].append(rec_len)
        rows["tier"].append("iphi"); rows["clean"].append(1.0); rows["dup_frac"].append(0.0)
        if with_meta:
            rows["region_id"].append(i % 3)
            rows["century_id"].append(i % 5)
        off += rec_len
    for p in PLANES:
        np.concatenate(bufs[p]).tofile(d / f"{p}.bin")
    pq.write_table(pa.table(rows), d / "index.parquet")


def test_shardset_reads_region_century_columns(tmp_path):
    d = tmp_path / "shard_with_meta"
    _write_shard(d, with_meta=True)
    ss = ShardSet(str(d))
    assert list(ss.region_id[:3]) == [0, 1, 2]
    assert list(ss.century_id[:5]) == [0, 1, 2, 3, 4]


def test_shardset_defaults_unk_when_columns_absent(tmp_path):
    """Old shards (GCB pretraining, gold/silver/bronze) never had these columns."""
    d = tmp_path / "shard_no_meta"
    _write_shard(d, with_meta=False)
    ss = ShardSet(str(d))
    assert (ss.region_id == UNK_REGION).all()
    assert (ss.century_id == UNK_CENTURY).all()


def test_multitier_loader_yields_region_century_and_survives_windowing(tmp_path):
    d = tmp_path / "shard_windowed"
    _write_shard(d, n_records=4, rec_len=5000, with_meta=True)  # long enough to force _window
    cfg = DataConfig(tiers={"iphi": TierSpec(str(d), 1.0, tier_filter="iphi")},
                     window_chars=256, seed=0)
    loader = MultiTierLoader(cfg)
    recs = list(loader.records(20))
    assert recs
    for r in recs:
        assert len(r["chars"]) <= 256
        assert isinstance(r["region_id"], (int, np.integer))
        assert isinstance(r["century_id"], (int, np.integer))
        assert r["region_id"] in (0, 1, 2)
        assert r["century_id"] in (0, 1, 2, 3, 4)
