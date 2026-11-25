"""Load the Norma macronization/scansion benchmark.

Two sources:
  hf  (default) -- the canonical, actively-maintained release at
      https://huggingface.co/datasets/Ericu950/norma. This now supersedes the old
      git-based copy: at time of writing it has 1378 macronize rows (614 test +
      764 dev) vs the git copy's 932 (both dev and test folded into one file,
      disambiguated by split_manifest.json), plus the same 614 syllabify test rows.
  git -- a local checkout of an existing macronization project ($MACRONIZER_SRC),
      kept for exact reproducibility of numbers reported before the HF migration.

Either way, load_norma() returns the same shape: {"test": [...], "dev": [...]},
each item a {"text": ..., "source": ..., "task": "macronize"|"syllabify"} dict,
so callers don't need to know which source produced them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HF_REPO = "Ericu950/norma"


def _load_git(src: Path) -> dict:
    lines = [json.loads(l) for l in open(src / "data/norma/test.jsonl", encoding="utf-8")]
    manifest = json.loads((src / "data/norma/split_manifest.json").read_text())
    dev_idx = set(manifest["dev_line_indices"])
    test = [d for i, d in enumerate(lines) if i not in dev_idx]
    dev = [d for i, d in enumerate(lines) if i in dev_idx]
    return {"test": test, "dev": dev}


def _load_hf() -> dict:
    from huggingface_hub import hf_hub_download
    test_path = hf_hub_download(HF_REPO, "data/test.jsonl", repo_type="dataset")
    dev_path = hf_hub_download(HF_REPO, "data/validation.jsonl", repo_type="dataset")
    test = [json.loads(l) for l in open(test_path, encoding="utf-8")]
    dev = [json.loads(l) for l in open(dev_path, encoding="utf-8")]
    return {"test": test, "dev": dev}


def load_norma(source: str = "hf", src: Path | None = None) -> dict:
    """source="hf" (default): pull the canonical release from the Hub, no local
    clone needed. source="git": read src/data/norma/{test.jsonl,split_manifest.json}
    from a local macron-data checkout (src defaults to $MACRONIZER_SRC)."""
    if source == "hf":
        return _load_hf()
    if source == "git":
        src = src or Path(os.path.expandvars(os.environ.get(
            "MACRONIZER_SRC", "$MACRONIZER_SRC")))
        return _load_git(src)
    raise ValueError(f"unknown Norma source: {source!r} (expected 'hf' or 'git')")


def add_norma_source_arg(ap: argparse.ArgumentParser) -> None:
    """Shared CLI flag so every Norma-consuming entry point takes the same option."""
    ap.add_argument("--norma-source", choices=["hf", "git"], default="hf",
                    help="where to load the Norma benchmark from (default: hf)")
