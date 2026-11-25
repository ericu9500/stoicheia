"""Convert a raw Stoicheia training checkpoint ({model, opt, step, cfg}) into a clean,
HF-Hub-ready model repo (config.json + model.safetensors + modeling/configuration/
processing .py files + model card), dropping optimizer state and training-only config.

  python convert_checkpoint_to_hf.py --kind backbone \
      --ckpt $STOICHEIA_DATA/runs/stoicheia_doc_clean/best.pt \
      --out hf_release/Stoicheia-doc_clean \
      --name "Stoicheia (documentary-clean)" \
      --metrics-json $STOICHEIA_DATA/runs/stoicheia_doc_clean/eval.jsonl

  python convert_checkpoint_to_hf.py --kind tagger_parser \
      --ckpt $STOICHEIA_DATA/parser_data/runs/joint_docclean_f3_s0/best.pt \
      --out hf_release/Stoicheia-tagger-parser \
      --vocab-json $STOICHEIA_DATA/parser_data/runs/joint_docclean_f3_s0/vocab.json \
      --deprel-vocab $STOICHEIA_DATA/parser_data/runs/joint_docclean_f3_s0/deprel_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

# Architecture-only fields of CharBertConfig -- everything else in a raw checkpoint's
# "cfg" dict (lr, wd, tier_weights, anneal_phases, out_dir, total_steps, ...) is training
# metadata and is dropped from config.json (kept in a training_metadata.json sidecar).
_ARCH_FIELDS = [
    "n_alpha", "mask_id", "blank_id", "pad_id", "n_char_ids",
    "n_boundary", "n_dia", "n_punct", "d_model", "n_heads", "depth",
    "char_window", "attn_impl", "qk_norm",
]
_ARCH_DEFAULTS = dict(
    n_alpha=24, mask_id=24, blank_id=25, pad_id=26, n_char_ids=27,
    n_boundary=4, n_dia=49, n_punct=7, n_heads=16, char_window=256,
    attn_impl="sdpa", qk_norm=True,
)

_THIS_DIR = Path(__file__).resolve().parent.parent / "hf_release"


def _arch_config_from_raw_cfg(raw_cfg: dict) -> dict:
    """raw_cfg is the full training config embedded in a checkpoint (mixes architecture
    + training-only keys, sometimes missing fields that have a fixed architectural
    default). Extract just the CharBertConfig-shape subset."""
    out = dict(_ARCH_DEFAULTS)
    for k in _ARCH_FIELDS:
        if k in raw_cfg:
            out[k] = raw_cfg[k]
    # n_heads is derived as d_model // 64 throughout the training code, not always stored
    if "d_model" in raw_cfg:
        out["d_model"] = raw_cfg["d_model"]
        out.setdefault("n_heads", raw_cfg["d_model"] // 64)
        out["n_heads"] = raw_cfg.get("n_heads", raw_cfg["d_model"] // 64)
    if "attn" in raw_cfg and "attn_impl" not in raw_cfg:
        out["attn_impl"] = "sdpa"  # publish with the portable path regardless of training-time attn
    return out


def _joint_config_from_ckpt(sd: dict) -> dict:
    """Build a CharBertJointConfig-shape dict for a JointModel checkpoint (tagger.* +
    biaffine.* state dict), reading every label-space size straight off real tensor shapes
    rather than trusting the training cfg dicts (which record loss weights, not architecture)."""
    state_dict = sd["model"]
    pretrain_cfg = sd.get("pretrain_cfg", {})
    tcfg = sd.get("tcfg", {})
    pcfg = sd.get("pcfg", {})

    d_model = state_dict["tagger.encoder.e_char.weight"].shape[1]
    depth = 1 + max(
        int(k.split(".")[3]) for k in state_dict if k.startswith("tagger.encoder.blocks."))
    n_xpos_classes = []
    p = 0
    while f"tagger.xpos_heads.{p}.weight" in state_dict:
        n_xpos_classes.append(state_dict[f"tagger.xpos_heads.{p}.weight"].shape[0])
        p += 1

    use_cap = "tagger.encoder.cap_emb.weight" in state_dict
    use_flat = "tagger.head_flat.weight" in state_dict
    n_script = state_dict["tagger.head_script.weight"].shape[0]
    n_upos = state_dict["tagger.head_upos.weight"].shape[0]
    n_flat_tags = state_dict["tagger.head_flat.weight"].shape[0] if use_flat else 0
    n_labels = state_dict["biaffine.rel_biaf.W"].shape[0]
    d_arc = state_dict["biaffine.arc_dep.lin.weight"].shape[0]
    d_rel = state_dict["biaffine.rel_dep.lin.weight"].shape[0]

    config = dict(
        n_alpha=24, mask_id=24, blank_id=25, pad_id=26, n_char_ids=27,
        n_boundary=4, n_dia=49, n_punct=7,
        d_model=d_model, n_heads=pretrain_cfg.get("d_model", d_model) // 64, depth=depth,
        char_window=pretrain_cfg.get("char_window", 256), attn_impl="sdpa",
        qk_norm=pretrain_cfg.get("qk_norm", True),
        use_cap=use_cap,
        pool=tcfg.get("pool", "mean"), head_dropout=tcfg.get("head_dropout", 0.1),
        scalar_mix=tcfg.get("scalar_mix", False), xpos_len=len(n_xpos_classes),
        n_xpos_classes=n_xpos_classes, n_script=n_script, n_upos=n_upos,
        use_flat=use_flat, n_flat_tags=n_flat_tags,
        d_arc=d_arc, d_rel=d_rel, n_labels=n_labels,
        parse_dropout=pcfg.get("dropout", 0.33),
        max_chars=sd.get("T", 2048), max_words=sd.get("W", 384),
    )
    return config


def _meter_config_from_ckpt(sd: dict) -> dict:
    """Build a CharBertMeterConfig-shape dict for a MeterModel checkpoint
    (encoder.* + head_mac/head_scan/mix_w state dict), reading architecture-defining
    fields off real tensor shapes / key presence rather than trusting mcfg (which
    also carries loss-only fields like head_dropout/class weights)."""
    state_dict = sd["model"]
    pretrain_cfg = sd.get("pretrain_cfg", {})

    d_model = state_dict["encoder.e_char.weight"].shape[1]
    depth = 1 + max(
        int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.blocks."))
    use_cap = "encoder.cap_emb.weight" in state_dict
    scalar_mix = "mix_w" in state_dict

    return dict(
        n_alpha=24, mask_id=24, blank_id=25, pad_id=26, n_char_ids=27,
        n_boundary=4, n_dia=49, n_punct=7,
        d_model=d_model, n_heads=pretrain_cfg.get("d_model", d_model) // 64, depth=depth,
        char_window=pretrain_cfg.get("char_window", 256), attn_impl="sdpa",
        qk_norm=pretrain_cfg.get("qk_norm", True),
        use_cap=use_cap, scalar_mix=scalar_mix,
    )


def convert(ckpt_path: str, out_dir: str, kind: str, vocab_json: str | None = None,
            deprel_vocab: str | None = None, name: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = sd["model"]

    if kind == "tagger_parser":
        assert vocab_json and deprel_vocab, "tagger_parser conversion needs --vocab-json and --deprel-vocab"
        assert "tagger.encoder.e_char.weight" in state_dict and "biaffine.root" in state_dict, \
            f"unexpected checkpoint shape for kind=tagger_parser: {sorted(state_dict.keys())[:5]}..."

        config = _joint_config_from_ckpt(sd)
        config["model_type"] = "char_bert_joint"
        config["auto_map"] = {
            "AutoConfig": "configuration_char_bert_joint.CharBertJointConfig",
            "AutoModel": "modeling_char_bert_joint.CharBertForTaggingAndParsing",
        }
        (out / "config.json").write_text(json.dumps(config, indent=2))

        training_meta = {
            "cfg": sd.get("cfg"), "tcfg": sd.get("tcfg"), "pcfg": sd.get("pcfg"),
            "pretrain_cfg": sd.get("pretrain_cfg"), "epoch": sd.get("epoch"), "dev": sd.get("dev"),
            "_source_checkpoint": Path(ckpt_path).name,
        }
        (out / "training_metadata.json").write_text(json.dumps(_scrub_paths(training_meta), indent=2, default=str))

        clean_sd = {k: v.contiguous() for k, v in state_dict.items()}
        save_file(clean_sd, str(out / "model.safetensors"))

        for fname in ("configuration_char_bert_joint.py", "modeling_char_bert_joint.py",
                      "processing_char_bert_joint.py"):
            shutil.copy(_THIS_DIR / fname, out / fname)
        shutil.copy(vocab_json, out / "vocab.json")
        shutil.copy(deprel_vocab, out / "deprel_vocab.json")

        print(f"converted {ckpt_path} -> {out}  (kind={kind}, d_model={config['d_model']}, "
              f"depth={config['depth']}, n_labels={config['n_labels']}, "
              f"params={sum(v.numel() for v in clean_sd.values()):,})")
        return out

    if kind == "meter":
        assert "encoder.e_char.weight" in state_dict and "head_mac.weight" in state_dict, \
            f"unexpected checkpoint shape for kind=meter: {sorted(state_dict.keys())[:5]}..."

        config = _meter_config_from_ckpt(sd)
        config["model_type"] = "char_bert_meter"
        config["auto_map"] = {
            "AutoConfig": "configuration_char_bert_meter.CharBertMeterConfig",
            "AutoModel": "modeling_char_bert_meter.CharBertMeterModel",
        }
        (out / "config.json").write_text(json.dumps(config, indent=2))

        training_meta = {
            "cfg": sd.get("cfg"), "mcfg": sd.get("mcfg"), "pretrain_cfg": sd.get("pretrain_cfg"),
            "epoch": sd.get("epoch"), "dev": sd.get("dev"), "T": sd.get("T"),
            "_source_checkpoint": Path(ckpt_path).name,
        }
        (out / "training_metadata.json").write_text(json.dumps(_scrub_paths(training_meta), indent=2, default=str))

        # mac_w/scan_w are loss-only class weights (used only by MeterModel.loss(),
        # never by forward()); the HF wrapper doesn't declare them, so drop them here
        # instead of shipping dead buffers alongside an inference-only model
        clean_sd = {k: v.contiguous() for k, v in state_dict.items() if k not in ("mac_w", "scan_w")}
        save_file(clean_sd, str(out / "model.safetensors"))

        for fname in ("configuration_char_bert_meter.py", "modeling_char_bert_meter.py",
                      "processing_char_bert_meter.py"):
            shutil.copy(_THIS_DIR / fname, out / fname)

        print(f"converted {ckpt_path} -> {out}  (kind={kind}, d_model={config['d_model']}, "
              f"depth={config['depth']}, params={sum(v.numel() for v in clean_sd.values()):,})")
        return out

    assert "model" in sd and "cfg" in sd, f"unexpected checkpoint shape: {list(sd.keys())}"
    raw_cfg = sd["cfg"]

    arch_cfg = _arch_config_from_raw_cfg(raw_cfg)

    # sanity check: head_char is not weight-tied to e_char (separate matrices)
    if "e_char.weight" in state_dict and "head_char.weight" in state_dict:
        assert state_dict["e_char.weight"].data_ptr() != state_dict["head_char.weight"].data_ptr(), \
            "unexpected weight tying between e_char and head_char -- conversion assumes untied weights"

    config = dict(arch_cfg)
    config["model_type"] = "char_bert"
    config["auto_map"] = {
        "AutoConfig": "configuration_char_bert.CharBertConfig",
        "AutoModel": "modeling_char_bert.CharBertModel",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # training-only metadata, kept for provenance/appendix purposes, not needed to load the model
    training_meta = {k: v for k, v in raw_cfg.items() if k not in _ARCH_FIELDS}
    training_meta["_source_checkpoint"] = Path(ckpt_path).name
    training_meta["_source_step"] = sd.get("step")
    (out / "training_metadata.json").write_text(json.dumps(_scrub_paths(training_meta), indent=2, default=str))

    # weights: drop optimizer state, keep only the model's own state dict
    clean_sd = {k: v.contiguous() for k, v in state_dict.items()}
    save_file(clean_sd, str(out / "model.safetensors"))

    for fname in ("configuration_char_bert.py", "modeling_char_bert.py", "processing_char_bert.py"):
        shutil.copy(_THIS_DIR / fname, out / fname)

    print(f"converted {ckpt_path} -> {out}  (kind={kind}, d_model={arch_cfg['d_model']}, "
          f"depth={arch_cfg['depth']}, params={sum(v.numel() for v in clean_sd.values()):,})")
    return out


def _scrub_paths(obj):
    """Absolute cluster paths in a training cfg would identify the machine (and its owner),
    so reduce every path-like value to its basename before it reaches the sidecar."""
    import re as _re
    if isinstance(obj, dict):
        return {k: _scrub_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_paths(v) for v in obj]
    if isinstance(obj, str):
        if ("/" in obj) and _re.search(r"^(/|\$|~)", obj):
            obj = obj.rsplit("/", 1)[-1]
        # run directory names are training bookkeeping, not part of the release's
        # vocabulary; keep only the trailing component so the sidecar stays neutral
        obj = obj.rsplit("/", 1)[-1]
        return obj
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["backbone", "restoration", "tagger_parser", "meter"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-json", default=None)
    ap.add_argument("--deprel-vocab", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    convert(os.path.expandvars(a.ckpt), a.out, a.kind,
            vocab_json=os.path.expandvars(a.vocab_json) if a.vocab_json else None,
            deprel_vocab=os.path.expandvars(a.deprel_vocab) if a.deprel_vocab else None,
            name=a.name)


if __name__ == "__main__":
    main()
