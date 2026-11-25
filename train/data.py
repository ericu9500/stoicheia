"""3-tier streaming data loader over memmap shards (plan §1, §6 curriculum).

Samples records from gold/silver/bronze with configurable tier weights and a cleanliness
floor, yielding fixed-T char windows ready for the collator. Two phases:
  stable  weights = {gold, silver, bronze} (bronze included for coverage)
  anneal  weights = {gold: 1, silver: 0, bronze: 0}  (gold-only, washes out synth/repair bias)

Long records are chopped into T-char windows on word boundaries; short ones are used whole
(the collator packs several per row). Deterministic given (seed, rank, world_size) so runs
resume exactly and DP ranks see disjoint data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Metadata-conditioning UNK ids (insc/data/meta_vocab.py's UNK_REGION/UNK_CENTURY) -- same
# hardcoded-constant convention as train/collate.py's UNK_REGION/UNK_CENTURY, for the same
# reason (this is base-pretraining-shared code; the insc-only meta_vocab package isn't
# always on sys.path here). Shards without region_id/century_id columns (every GCB
# pretraining/gold/silver/bronze shard, built before this existed) fall back to these.
UNK_REGION, UNK_CENTURY = 14, 15


@dataclass
class TierSpec:
    path: str
    weight: float
    tier_filter: str = None      # keep only records whose `tier` column == this (None = all)


@dataclass
class DataConfig:
    tiers: dict = field(default_factory=dict)     # name -> TierSpec
    min_clean: float = 0.0
    drop_dup_frac: float = 0.7
    window_chars: int = 4096
    seed: int = 0
    exclude_holdout: bool = True   # False = train on all records (external dev set drives eval)


class ShardSet:
    """One shard directory's memmap planes + record index (memmaps shared across tiers)."""
    _cache = {}
    _warned = set()

    def __init__(self, path):
        d = Path(path)
        ip = d / "index_dedup.parquet"
        if not ip.exists() and path not in ShardSet._warned:
            # the near-duplicate filter (dup_frac < drop_dup_frac) reads index_dedup.parquet
            import warnings
            warnings.warn(f"{d}: no index_dedup.parquet — training WITHOUT the near-duplicate filter")
            ShardSet._warned.add(path)
        idx = pq.read_table(ip if ip.exists() else d / "index.parquet")
        self.offset = idx.column("offset").to_numpy()
        self.length = idx.column("length").to_numpy()
        self.clean = idx.column("clean").to_numpy()
        self.tier = idx.column("tier").to_numpy(zero_copy_only=False)
        self.dup = (idx.column("dup_frac").to_numpy() if "dup_frac" in idx.column_names
                    else np.zeros(len(self.offset), np.float32))
        n = len(self.offset)
        self.region_id = (idx.column("region_id").to_numpy() if "region_id" in idx.column_names
                          else np.full(n, UNK_REGION, np.int64))
        self.century_id = (idx.column("century_id").to_numpy() if "century_id" in idx.column_names
                           else np.full(n, UNK_CENTURY, np.int64))
        self.chars = np.memmap(d / "chars.bin", dtype=np.uint8, mode="r")
        self.boundary = np.memmap(d / "boundary.bin", dtype=np.uint8, mode="r")
        self.dia = np.memmap(d / "dia.bin", dtype=np.uint8, mode="r")
        self.cap = np.memmap(d / "cap.bin", dtype=np.uint8, mode="r")
        punct_p = d / "punct.bin"
        # tolerate shards built before the punctuation plane existed: falls back to all-"none"
        self.punct = np.memmap(punct_p, dtype=np.uint8, mode="r") if punct_p.exists() else None

    @classmethod
    def get(cls, path):
        if path not in cls._cache:
            cls._cache[path] = cls(path)
        return cls._cache[path]

    def eligible(self, min_clean, drop_dup, tier_filter=None, exclude_holdout=True):
        m = (self.clean >= min_clean) & (self.dup < drop_dup)
        if tier_filter is not None:
            m = m & (self.tier == tier_filter)
        idx = np.flatnonzero(m)
        if exclude_holdout:                       # reserve every HOLDOUT_MOD-th record for eval
            idx = idx[idx % HOLDOUT_MOD != 0]
        return idx


HOLDOUT_MOD = 200   # ~0.5% held out from training; eval/intrinsic selects idx % HOLDOUT_MOD == 0


class MultiTierLoader:
    def __init__(self, cfg: DataConfig, rank=0, world_size=1):
        self.cfg = cfg
        self.rank, self.world = rank, world_size
        self.sets, self.elig, self.names, self.wts = {}, {}, [], []
        for name, spec in cfg.tiers.items():
            if spec.weight <= 0:
                continue
            ss = ShardSet.get(spec.path)
            el = ss.eligible(cfg.min_clean, cfg.drop_dup_frac, spec.tier_filter,
                             exclude_holdout=cfg.exclude_holdout)
            # shard eligible records across DP ranks
            el = el[rank::world_size]
            if len(el) == 0:
                continue
            self.sets[name] = ss
            self.elig[name] = el
            self.names.append(name)
            self.wts.append(spec.weight)
        assert self.names, "no eligible tiers"
        self.wts = np.array(self.wts, float)
        self.wts /= self.wts.sum()
        self.rng = np.random.default_rng(cfg.seed + 1315423911 * rank)
        self._cursor = {n: 0 for n in self.names}
        self._perm = {n: self.rng.permutation(self.elig[n]) for n in self.names}

    def _next_record(self, name):
        ss = self.sets[name]
        p = self._perm[name]
        c = self._cursor[name]
        if c >= len(p):
            self._perm[name] = self.rng.permutation(self.elig[name])
            c = 0
        i = int(self._perm[name][c])
        self._cursor[name] = c + 1
        o, l = int(ss.offset[i]), int(ss.length[i])
        punct = (np.asarray(ss.punct[o:o+l]) if ss.punct is not None
                else np.zeros(l, dtype=np.uint8))
        return dict(chars=np.asarray(ss.chars[o:o+l]), boundary=np.asarray(ss.boundary[o:o+l]),
                    dia=np.asarray(ss.dia[o:o+l]), cap=np.asarray(ss.cap[o:o+l]), punct=punct,
                    region_id=int(ss.region_id[i]), century_id=int(ss.century_id[i]))

    def _window(self, rec):
        """Chop a long record to <= window_chars on a word boundary; else return whole."""
        W = self.cfg.window_chars
        n = len(rec["chars"])
        if n <= W:
            return rec
        b = rec["boundary"]
        start = int(self.rng.integers(0, n - W))
        # snap start to just after a boundary, end to a boundary
        we = np.flatnonzero(b[:start] >= 1)
        s = (we[-1] + 1) if len(we) else 0
        seg_end = np.flatnonzero(b[s:s+W] >= 1)
        e = (s + seg_end[-1] + 1) if len(seg_end) else min(s + W, n)
        # per-character planes get windowed; scalar per-record metadata (region_id/
        # century_id) passes through unchanged -- it describes the whole inscription/
        # papyrus, not any one character span within it.
        return {k: (v[s:e] if isinstance(v, np.ndarray) else v) for k, v in rec.items()}

    def records(self, n):
        """Yield n windowed records sampled by tier weight."""
        picks = self.rng.choice(len(self.names), size=n, p=self.wts)
        for pi in picks:
            yield self._window(self._next_record(self.names[pi]))


def stable_cfg(gdata, w=(1.0, 1.0, 0.3), window=4096, seed=0, exclude_holdout=True):
    """gold:silver:bronze default weights (bronze down-weighted as synthetic)."""
    return DataConfig(tiers={
        "gold":   TierSpec(f"{gdata}/shards/v1_punct", w[0], tier_filter="pristine"),
        "silver": TierSpec(f"{gdata}/shards/v1_punct", w[1], tier_filter="repaired"),
        "bronze": TierSpec(f"{gdata}/shards/bronze_punct", w[2], tier_filter="bronze"),
    }, window_chars=window, seed=seed, exclude_holdout=exclude_holdout)


def anneal_cfg(gdata, window=4096, seed=0, exclude_holdout=True):
    """gold-only anneal phase."""
    return DataConfig(tiers={
        "gold": TierSpec(f"{gdata}/shards/v1_punct", 1.0, tier_filter="pristine"),
    }, window_chars=window, seed=seed, exclude_holdout=exclude_holdout)
