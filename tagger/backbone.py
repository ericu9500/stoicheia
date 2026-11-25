"""Bridge to the Stoicheia (Stoicheia) backbone — the ONLY module that touches it.

The pretraining repo is imported live via $STOICHEIA_ROOT (see env.sh); nothing there is
modified. At release time this shim is the single place to swap in the published
Stoicheia package or a vendored copy of model/ + data/normalize.py.
"""
from __future__ import annotations

import os
import sys

_GCB = os.path.expandvars(os.environ.get("STOICHEIA_ROOT", "$STOICHEIA_DATA"))
if _GCB not in sys.path:
    sys.path.insert(0, _GCB)

import torch

from data.normalize import (  # noqa: F401  (re-exported for the rest of the package)
    ALPHABET, DIA_STATES, N_PUNCT, Stats, normalize_record, restore_polytonic, unpack_dia,
)
from model.char_bert import CharBertConfig, CharBertEncoder

UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6
PAD_ID = 26


class CharBertWithHidden(CharBertEncoder):
    """CharBertEncoder whose forward also returns the final hidden state.

    forward() is a verbatim copy of the parent's with `hidden=x` added to the output —
    the parent discards x after the output heads. State dict is identical, so
    pretraining checkpoints load strictly.
    """

    def forward(self, batch):
        from model.layers import build_attn_mask, build_block_mask

        cfg = self.cfg
        ids = batch["input_ids"]
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        seg = batch["seg_id"]

        x = (self.e_char(ids) + self.e_bnd(batch["boundary"]) + self.e_dia(batch["dia"])
             + self.e_punct(batch["punct"]))
        # optional fine-tune-only capitalization channel (pretraining treats cap as
        # output-only); zero-init so loading a pretraining checkpoint is a no-op
        cap_emb = getattr(self, "cap_emb", None)
        if cap_emb is not None and "cap" in batch:
            x = x + cap_emb(batch["cap"])

        if cfg.attn_impl == "flex":
            char_mask = build_block_mask(seg, cfg.char_window, ids.device)
            glob_mask = build_block_mask(seg, 0, ids.device)
        else:
            char_mask = build_attn_mask(seg, cfg.char_window, ids.device, x.dtype)
            glob_mask = build_attn_mask(seg, 0, ids.device, x.dtype)

        collect = getattr(self, "return_layers", False)
        layers = []
        for blk in self.blocks:
            m = glob_mask if blk.window == 0 else char_mask
            x = blk(x, pos, m)
            if collect:
                layers.append(x)

        x = self.norm_out(x)
        return dict(
            layers=layers,
            hidden=x,
            char=self.head_char(x),
            boundary=self.head_bnd(x),
            dia=self.head_dia(x),
            cap=self.head_cap(x),
            punct=self.head_punct(x),
        )


def load_backbone(ckpt_path, device, attn_impl="sdpa"):
    """Load a Stoicheia pretraining checkpoint into a hidden-exposing encoder.

    Mirrors eval/intrinsic.py::load_model; returns (model, pretrain_cfg_dict).
    """
    # Pretraining ablation: "random:<ckpt>" builds the SAME architecture as <ckpt> but
    # leaves the weights at their random init, so capacity/tokenisation/finetune recipe are
    # identical and the metric delta is exactly what pretraining contributed. Mirrors
    # meter/backbone.py::load_backbone.
    # representation ablation: flat-character / BPE token checkpoints are served
    # through the plane interface (ablate/tok_model.py::TokenAsPlanes)
    from ablate.tok_model import load_any_backbone
    tok_backbone = load_any_backbone(ckpt_path, device, attn_impl=attn_impl)
    if tok_backbone is not None:
        return tok_backbone
    random_init = False
    ckpt_path = str(ckpt_path)
    if ckpt_path.startswith("random:"):
        random_init = True
        ckpt_path = ckpt_path.split(":", 1)[1]
    sd = torch.load(os.path.expandvars(ckpt_path), map_location="cpu", weights_only=False)
    c = sd["cfg"]
    mcfg = CharBertConfig(attn_impl=attn_impl, d_model=c["d_model"],
                          n_heads=c["d_model"] // 64, depth=c["depth"],
                          char_window=c["char_window"], qk_norm=c.get("qk_norm", True))
    model = CharBertWithHidden(mcfg)
    if random_init:
        print(f"RANDOM-INIT backbone (architecture from {ckpt_path}, weights NOT loaded)",
              flush=True)
    else:
        model.load_state_dict(sd["model"])
    return model.to(device), c


def load_backbone_auto(cfg, device):
    """Dispatch on cfg['backbone']: 'charbert' (default; existing torch.save checkpoint via
    load_backbone above) or 'hf' (a HuggingFace hub id or local directory, given as
    cfg['hf_model_name_or_path'], loaded through tagger.hf_backbone.load_hf_backbone).

    Used by both tagger/train.py and parser/joint_train.py so a training config can request
    either encoder family without either script special-casing the two paths beyond this call.

    Returns (encoder, pretrain_cfg_dict, tokenizer_or_None) -- tokenizer is only non-None for
    the 'hf' path; pass it straight through to TaggerDataset(..., tokenizer=tokenizer) to switch
    the dataset onto the HF subword batching path.
    """
    kind = cfg.get("backbone", "charbert")
    if kind == "charbert":
        encoder, pcfg = load_backbone(cfg["ckpt"], device, attn_impl=cfg.get("attn", "sdpa"))
        return encoder, pcfg, None
    if kind == "hf":
        from tagger.hf_backbone import load_hf_backbone
        encoder, pcfg, tokenizer = load_hf_backbone(
            cfg["hf_model_name_or_path"], device,
            add_prefix_space=cfg.get("hf_add_prefix_space", True))
        return encoder, pcfg, tokenizer
    raise ValueError(f"tagger.backbone.load_backbone_auto: unknown backbone kind {kind!r}")
