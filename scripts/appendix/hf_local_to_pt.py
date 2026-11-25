"""Convert a locally snapshotted released checkpoint (config.json + model.safetensors) to the
trainer .pt format — same logic as scripts/hf_to_eval_checkpoint.py, but offline."""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
ARCH = ["n_alpha", "mask_id", "blank_id", "pad_id", "n_char_ids", "n_boundary", "n_dia", "n_punct",
        "d_model", "n_heads", "depth", "char_window", "attn_impl", "qk_norm", "use_cap", "scalar_mix"]
src, out = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.load(open(src / "config.json"))
sd = load_file(str(src / "model.safetensors"))
sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": sd, "cfg": {k: cfg[k] for k in ARCH if k in cfg}, "step": cfg.get("_step"),
            "_source": str(src)}, out)
print(f"wrote {out} ({sum(v.numel() for v in sd.values())/1e6:.1f}M params, {len(sd)} tensors)")
