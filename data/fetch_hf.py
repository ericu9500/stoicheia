"""Download the two source datasets from HuggingFace and record exact revisions.

Everything the shards are built from comes through here, so a rebuild from a clean
checkout + this script reproduces the corpus byte-for-byte (provenance.json pins the
dataset commit SHAs that were actually downloaded).

  python data/fetch_hf.py --out $STOICHEIA_DATA/raw [--revision-gold SHA] [--revision-bronze SHA]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

GOLD_REPO = "Ericu950/AncientGreek"                              # pristine + repaired parquet
BRONZE_REPO = "Ericu950/SyntheticAncientGreek-CorpusCorporum"    # bronze.jsonl (synthetic)


def fetch(repo_id, dest, revision=None):
    api = HfApi()
    sha = revision or api.dataset_info(repo_id).sha
    path = snapshot_download(repo_id, repo_type="dataset", revision=sha, local_dir=dest)
    print(f"{repo_id} @ {sha} -> {path}")
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision-gold", default=None, help="pin a specific commit (default: latest)")
    ap.add_argument("--revision-bronze", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    prov = {
        "gold_silver": {"repo": GOLD_REPO,
                        "revision": fetch(GOLD_REPO, out / "AncientGreek", a.revision_gold)},
        "bronze": {"repo": BRONZE_REPO,
                   "revision": fetch(BRONZE_REPO, out / "SyntheticAncientGreek-CorpusCorporum",
                                     a.revision_bronze)},
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"wrote {out / 'provenance.json'}")


if __name__ == "__main__":
    main()
