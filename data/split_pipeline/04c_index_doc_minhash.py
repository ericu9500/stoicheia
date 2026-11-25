#!/usr/bin/env python3
"""Stage 4c (documentary-clean variant): whole-RECORD MinHash signatures for every
documentary record (papyri any digit + PHI inscription real-edition variants, same
population as stage 4b), for a document-level near-duplicate safety net.

Why this exists: 05b_masks_doc.py's sentence-level exact/bag/8-gram matching (stage
4b's index) assumes "documentary editions are all indexed directly, near-verbatim
quotes are caught by the 8-gram layer" -- but scanned SOURCEBOOK volumes (Dittenberger's
Sylloge Inscriptionum Graecarum, Schwyzer's Dialectorum Graecarum Exempla Epigraphica
Potiora, Cagnat's Inscriptiones Graecae ad Res Romanas Pertinentes, and similar
epigraphic/papyrological corpora catalogued as ordinary "literary" books in the IA tier)
reproduce documentary text with enough OCR noise, editorial apparatus, and formatting
drift that individual 8-grams can miss while the DOCUMENT is still substantially the
same content. This mirrors the cross-edition safety net stage 5 already has for
literary-vs-literary duplicates (minhash_pristine.npz) -- here applied documentary-vs-
literary/bronze, at the document level.

Output: work/doc_clean/minhash_documentary.npz  (sigs [n,128] uint64, nwords [n])
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import h64, NGRAM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "work", "doc_clean")
PAPYRI_SOURCES = {"ddbdp", "dclp"}
INSCR_SYNTH = {"synthetic", "synthetic_2"}

N_PERM = 128
MERSENNE61 = np.uint64((1 << 61) - 1)
rng = np.random.RandomState(20260709)          # same perms as stage 2 / stage 5
PERM_A = rng.randint(1, 1 << 28, size=N_PERM).astype(np.uint64)
PERM_B = rng.randint(0, 1 << 32, size=N_PERM).astype(np.uint64)


def minhash(shingle_hashes):
    if len(shingle_hashes) == 0:
        return np.zeros(N_PERM, dtype=np.uint64)
    h = np.asarray(shingle_hashes, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    v = (PERM_A[:, None] * h[None, :] + PERM_B[:, None]) % MERSENNE61
    return v.min(axis=1)


def process_shard(args):
    tier, path = args
    t = pq.read_table(path, columns=["rid", "source", "skels"])
    sigs, nwords = [], []
    for rid, source, skels in zip(t["rid"].to_pylist(), t["source"].to_pylist(),
                                  t["skels"].to_pylist()):
        if tier == "inscriptions":
            field = rid.split(":", 1)[1] if ":" in rid else ""
            if field in INSCR_SYNTH:
                continue
        elif source not in PAPYRI_SOURCES:
            continue
        words_all = []
        for sk in skels:
            words_all.extend(sk.split())
        grams = ([h64(" ".join(words_all[i:i + NGRAM]))
                 for i in range(len(words_all) - NGRAM + 1)]
                if len(words_all) >= NGRAM else [])
        sigs.append(minhash(grams))
        nwords.append(len(words_all))
    return (np.array(sigs, dtype=np.uint64).reshape(-1, N_PERM)
            if sigs else np.zeros((0, N_PERM), np.uint64)), np.array(nwords, np.int32)


def main():
    tasks = []
    for tier in ("pristine", "repaired", "inscriptions"):
        for p in sorted(glob.glob(os.path.join(ROOT, "work", "sentences",
                                               tier, "shard_*.parquet"))):
            tasks.append((tier, p))
    print("%d shards" % len(tasks), flush=True)

    workers = max(4, min(16, (os.cpu_count() or 12) - 8))
    all_sigs, all_nwords = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sigs, nwords in ex.map(process_shard, tasks, chunksize=1):
            if len(nwords):
                all_sigs.append(sigs); all_nwords.append(nwords)
    sigs = np.concatenate(all_sigs, axis=0)
    nwords = np.concatenate(all_nwords, axis=0)
    np.savez(os.path.join(OUT, "minhash_documentary.npz"), sigs=sigs, nwords=nwords)
    print("documentary minhash records: %d" % len(nwords), flush=True)


if __name__ == "__main__":
    main()
