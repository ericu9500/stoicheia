#!/usr/bin/env python3
"""Turn a released Hub checkpoint back into the format the evaluation scripts read.

The Hub ships `model.safetensors` + `config.json` for `AutoModel.from_pretrained`.
`insc/eval/restore_strict.py` and its papyrus twin instead take `--ckpt <file>.pt`
holding `{"model": state_dict, "cfg": {...}}`, which is what training writes. This
converts the former into the latter so the paper's tables can be reproduced from the
public artifacts without writing an adapter.

  python scripts/hf_to_eval_checkpoint.py \
      --repo Ericu950/Stoicheia-restoration-test3 \
      --out $STOICHEIA_DATA/insc_data/runs/whole_v4_t3v4/best.pt

Then, with INSC_TEST_DIGIT/INSC_VAL_DIGIT matching the checkpoint's held-out digit:

  python -m insc.eval.restore_strict --ckpt <that file> \
      --samples insc/eval/frozen/strict_test_fold3_samples.json --out strict.json

The evaluation also needs the documentary corpus at $INS_DATA/raw/iphi.jsonl (see
REPRODUCING.md); the frozen sample file fixes which gaps are scored, so the numbers
do not depend on how that corpus is shuffled.
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# config.json carries the architecture only; the trainer's cfg dict is what the eval
# scripts consult, and these are the fields they actually read.
_ARCH = ["n_alpha", "mask_id", "blank_id", "pad_id", "n_char_ids", "n_boundary",
         "n_dia", "n_punct", "d_model", "n_heads", "depth", "char_window",
         "attn_impl", "qk_norm", "use_cap", "scalar_mix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Hub repo id of a released checkpoint")
    ap.add_argument("--out", required=True, help="path to write the .pt to")
    ap.add_argument("--revision", default=None, help="pin a specific revision")
    a = ap.parse_args()

    cfg = json.load(open(hf_hub_download(a.repo, "config.json", revision=a.revision)))
    sd = load_file(hf_hub_download(a.repo, "model.safetensors", revision=a.revision))

    # the HF wrapper prefixes the backbone; the trainer's state dict does not
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd,
                "cfg": {k: cfg[k] for k in _ARCH if k in cfg},
                "step": cfg.get("_step"),
                "_source": a.repo}, out)
    print(f"wrote {out}  ({sum(v.numel() for v in sd.values()) / 1e6:.1f}M parameters)")


if __name__ == "__main__":
    main()
