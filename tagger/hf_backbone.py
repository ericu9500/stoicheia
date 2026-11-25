"""HF subword-encoder backbone (AutoModel/AutoTokenizer) plugged into the SAME recipe
CharBertWithHidden feeds (tagger/backbone.py): factored XPOS heads, edit-script lemmatizer,
UPOS aux head, ELMo-style scalar mix over all layers, and (via the joint model) the biaffine
dependency parser. This is the "swap the encoder, keep everything downstream identical" path
for a same-recipe cross-encoder ablation (XLM-R base/large, GreBERTa, AncientGreekBERT, Logion,
and -- as encoder-only extractions, see below -- the T5 encoder-decoder checkpoints GreTa/PhilTa).

Contract this file must match exactly (see CharBertWithHidden.forward in tagger/backbone.py):
  - encoder.cfg.d_model                      (TaggerModel/JointModel read this)
  - encoder.blocks                           (layerwise-LR-decay walk + mix_w sizing)
  - encoder.return_layers  (bool, settable)  (TaggerModel flips this on for scalar_mix)
  - encoder.forward(batch) -> dict(layers=[...], hidden=(B,T,D))
      len(layers) == len(encoder.blocks) whenever return_layers is True; ignored otherwise.

Design decisions that deviate from a raw char encoder (flagged explicitly, not silently):
  - `.blocks` is the HF model's real transformer-layer ModuleList (`model.encoder.layer` for
    every BERT/RoBERTa/XLM-R-style AutoModel checked here -- GreBerta, Ancient-Greek-BERT,
    Logion -- or `model.encoder.block` for the T5Stack used by T5EncoderModel -- GreTa, PhilTa).
    `_find_layer_list` tries a short list of known attribute paths and raises a clear error if
    none match, rather than guessing silently for an unseen architecture.
  - `output_hidden_states=True` returns L+1 hidden states: [embeddings, layer_1, ..., layer_L].
    CharBertWithHidden's contract needs `layers` to have exactly `len(blocks) == L` entries
    plus one more distinct `hidden`. There is no extra final-norm transform sitting on top of
    the last block in a stock encoder-only HF model (unlike CharBERT's separate `norm_out`), so
    the closest faithful analogue is: `layers = hidden_states[:-1]` (embeddings + the first L-1
    block outputs -- L entries) and `hidden = hidden_states[-1]` (the true final block output).
    Every distinct hidden state the model produces is therefore used exactly once as a mix
    component, and the non-scalar-mix default (`out["hidden"]`) is genuinely the last layer --
    nothing is duplicated or dropped. T5Stack (checked directly against its HF source) collects
    `all_hidden_states` with the exact same bookkeeping -- one entry appended before each block,
    plus one final entry that IS post-`final_layer_norm` -- so this contract holds verbatim for
    the T5-encoder-only path too; no special-casing needed in `forward()` itself.
  - GreTa/PhilTa (bowphs/GreTa, bowphs/PhilTa) are T5 encoder-decoder checkpoints. This ablation
    never uses T5's native generative/seq2seq decoding for anything (not lemmatization, not
    tagging) -- the decoder is never even instantiated. `load_hf_backbone` detects `model_type
    == "t5"` from the config and loads `T5EncoderModel.from_pretrained(...)` instead of
    `AutoModel`: this builds ONLY `shared` (the embedding matrix) + `encoder` (a T5Stack) +
    `encoder.final_layer_norm`, silently dropping the `decoder.*`/`lm_head.*` weights on load.
    From that point on GreTa/PhilTa are just another `HFBackboneWithHidden`-wrapped encoder --
    same forward(), same word/subword alignment, same scalar mix, same tagger/lemma/UPOS/biaffine
    heads downstream -- with two config-shape differences absorbed only at load time:
      * T5Config has no `hidden_size`/`num_hidden_layers`; it uses `d_model`/`num_layers`
        instead (`_resolve_dims` below reads the right pair off `model_type`).
      * T5Stack has no `.embeddings` submodule (no learned positional embeddings at all -- T5
        uses relative-attention-bias parameters that live inside each block, so they already
        ride the normal per-block LLRD schedule); its tied input embedding is the top-level
        `.shared` matrix. `_find_embeddings` (used by `param_groups_hf`) checks for `.shared`
        as a fallback path so that lone matrix still gets the same shallowest-LR treatment
        BERT-style embeddings get, instead of silently falling into the flat-LR catch-all.
  - Word/subword alignment does NOT reuse Stoicheia's char-level normalize_record. It tokenizes
    the whole sentence (every token, Greek and non-Greek) with `is_split_into_words=True` and
    reads back `BatchEncoding.word_ids()`, which fast (Rust-backed) HF tokenizers support
    directly. See tagger/dataset.py::encode_sentence_hf for the alignment/labels code -- this
    module only owns encoder + tokenizer loading. Confirmed empirically for both T5TokenizerFast
    checkpoints (GreTa, PhilTa): word_ids() round-trips correctly.
  - RoBERTa/GPT2-BPE-style fast tokenizers refuse `is_split_into_words=True` unless constructed
    with `add_prefix_space=True` (confirmed empirically for bowphs/GreBerta); BERT/WordPiece
    fast tokenizers accept and silently ignore that kwarg (confirmed for Ancient-Greek-BERT and
    the local Logion checkpoint); PhilTa's T5TokenizerFast raises TypeError on it (confirmed
    empirically) and falls back cleanly to the no-kwarg load already handled below. So
    `add_prefix_space=True` is passed unconditionally, with a fallback to omitting it only if a
    given tokenizer class rejects the kwarg outright.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

# Attribute paths (tuples walked via getattr) known to hold the transformer layer ModuleList
# for AutoModel-loaded encoder-only checkpoints. Extend this list before trusting a new
# architecture -- _find_layer_list raises loudly rather than guessing.
_LAYER_PATHS = [
    ("encoder", "layer"),     # BERT / RoBERTa / XLM-RoBERTa / ELECTRA / Longformer, ...
    ("transformer", "layer"), # DistilBERT
    ("encoder", "layers"),    # a few newer encoder implementations spell it plural
    ("encoder", "block"),     # T5Stack (T5EncoderModel.encoder.block) -- GreTa, PhilTa
]

# Same idea for the (single) embedding module/matrix an encoder holds, used only by
# param_groups_hf to give embeddings the shallowest LLRD rate. Returns None (not a raise) when
# nothing matches -- callers fall back to the flat-LR catch-all, same as before this existed.
_EMB_PATHS = [
    ("embeddings",),  # BERT / RoBERTa / XLM-RoBERTa / ELECTRA / Longformer, ...
    ("shared",),      # T5EncoderModel's tied input embedding matrix (no positional-embedding
                      # module to go with it -- T5 uses relative-attention-bias params living
                      # inside encoder.block instead, so those already get the per-block LLRD
                      # rate the normal blocks loop assigns)
]


def _find_layer_list(hf_model):
    for path in _LAYER_PATHS:
        obj = hf_model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj) > 0:
            return obj
    raise RuntimeError(
        f"tagger.hf_backbone: could not find a transformer layer list on "
        f"{type(hf_model).__name__}; add its attribute path to _LAYER_PATHS.")


def _find_embeddings(hf_model):
    for path in _EMB_PATHS:
        obj = hf_model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


# model_type (from AutoConfig) -> HF class name that extracts JUST the encoder stack from an
# encoder-decoder checkpoint, never instantiating (or loading weights into) the decoder. Add an
# entry here before pointing this ablation at another seq2seq architecture (e.g. "mt5" ->
# "MT5EncoderModel"); anything not in this dict falls through to plain AutoModel, unchanged.
_ENCODER_ONLY_CLASSES = {
    "t5": "T5EncoderModel",   # bowphs/GreTa, bowphs/PhilTa
}


def _resolve_dims(cfg):
    """(d_model, n_layers) off a HF config, robust to T5Config's different attribute names.

    T5Config has no `hidden_size`/`num_hidden_layers` (its encoder and decoder stacks can differ
    in depth, so there's no single "num_hidden_layers"); it spells these `d_model`/`num_layers`
    (the latter is specifically the ENCODER stack's depth -- decoder depth is the separate
    `num_decoder_layers`, never read here since the decoder is never built on this path).
    """
    if getattr(cfg, "model_type", None) == "t5":
        return cfg.d_model, cfg.num_layers
    return cfg.hidden_size, cfg.num_hidden_layers


@dataclass
class HFConfig:
    d_model: int
    n_layers: int
    name_or_path: str


class HFBackboneWithHidden(nn.Module):
    """Wraps a HF AutoModel so it slots in wherever CharBertWithHidden does. See module
    docstring for the exact contract and the layer/hidden-state bookkeeping decision."""

    def __init__(self, hf_model, cfg: HFConfig):
        super().__init__()
        self.model = hf_model
        self.cfg = cfg
        self.return_layers = False   # flipped on by TaggerModel.__init__ when scalar_mix=True

    @property
    def blocks(self):
        return _find_layer_list(self.model)

    def forward(self, batch):
        out = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          output_hidden_states=True)
        hs = out.hidden_states                       # tuple, len == len(self.blocks) + 1
        hidden = hs[-1]
        layers = list(hs[:-1]) if self.return_layers else []
        return dict(layers=layers, hidden=hidden)


def load_hf_backbone(name_or_path, device, add_prefix_space=True):
    """Load a standard HF encoder (hub id or local directory) into a hidden-exposing wrapper.

    Mirrors tagger.backbone.load_backbone's return shape as closely as a different checkpoint
    format allows: (encoder, pretrain_cfg_dict, tokenizer). `tokenizer` is the extra piece (the
    CharBERT path has none -- normalize_record is a free function) that tagger/dataset.py needs
    to switch TaggerDataset onto the subword batching path.
    """
    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(name_or_path, add_prefix_space=add_prefix_space)
    except TypeError:
        # a handful of slow/legacy tokenizer classes don't accept add_prefix_space at all
        tok = AutoTokenizer.from_pretrained(name_or_path)
    if not tok.is_fast:
        raise RuntimeError(
            f"tagger.hf_backbone: {name_or_path} loaded a slow tokenizer; the HF backbone path "
            "needs a fast (Rust-backed) tokenizer for BatchEncoding.word_ids() alignment.")

    model_type = AutoConfig.from_pretrained(name_or_path).model_type
    enc_only_cls_name = _ENCODER_ONLY_CLASSES.get(model_type)
    if enc_only_cls_name is not None:
        # Encoder-decoder checkpoint (T5 family): extract the encoder stack ONLY. This never
        # builds, loads weights into, or runs the decoder -- no seq2seq generation happens
        # anywhere in this ablation. From here down GreTa/PhilTa are indistinguishable from any
        # other AutoModel encoder (see module docstring).
        enc_only_cls = getattr(transformers, enc_only_cls_name)
        hf_model = enc_only_cls.from_pretrained(name_or_path)
    else:
        hf_model = AutoModel.from_pretrained(name_or_path)
    d_model, n_layers = _resolve_dims(hf_model.config)
    cfg = HFConfig(d_model=d_model, n_layers=n_layers, name_or_path=str(name_or_path))
    encoder = HFBackboneWithHidden(hf_model, cfg).to(device)
    pretrain_cfg = dict(kind="hf", name_or_path=str(name_or_path),
                        hidden_size=cfg.d_model, num_hidden_layers=cfg.n_layers)
    return encoder, pretrain_cfg, tok


def param_groups_hf(model, cfg):
    """Layerwise-LR-decay param groups for an HF backbone -- the same convention as
    tagger.train.param_groups (decay only matrices, embeddings at the shallowest LR, LLRD walk
    up through encoder.blocks, heads at flat lr_head) but without any of the CharBERT-specific
    e_char/e_bnd/e_dia/e_punct embeddings or head_char/head_bnd/... pretraining-head freezing,
    which simply don't exist on a HF backbone."""
    llrd = cfg.get("llrd", 0.95)
    lr_enc, lr_head, wd = cfg["lr_enc"], cfg["lr_head"], cfg.get("wd", 0.01)
    enc = model.encoder
    blocks = enc.blocks
    depth = len(blocks)
    groups = {}

    def add(p, lr, is_enc=True):
        key = (lr, 0.0 if p.ndim < 2 else wd, is_enc)
        groups.setdefault(key, []).append(p)

    block_params = set()
    for blk in blocks:
        for p in blk.parameters():
            block_params.add(id(p))

    emb_lr = lr_enc * llrd ** depth
    emb_mod = _find_embeddings(enc.model)
    emb_params = set()
    if emb_mod is not None:
        for p in emb_mod.parameters():
            emb_params.add(id(p))
            add(p, emb_lr)

    for i, blk in enumerate(blocks):
        for p in blk.parameters():
            add(p, lr_enc * llrd ** (depth - 1 - i))

    # anything else under the HF backbone not covered above (e.g. a pooler head nobody uses
    # downstream) rides along at the plain encoder LR, same "harmless" convention as the
    # CharBERT path's pretraining-head groups
    for p in enc.model.parameters():
        if id(p) not in block_params and id(p) not in emb_params:
            add(p, lr_enc)

    heads = [model.xpos_heads, model.head_script, model.head_upos]
    if model.head_flat is not None:
        heads.append(model.head_flat)
    for m in heads:
        for p in m.parameters():
            add(p, lr_head, is_enc=False)
    if hasattr(model, "mix_w"):
        add(model.mix_w, lr_head, is_enc=False)
    cap_emb = getattr(enc, "cap_emb", None)
    if cap_emb is not None:
        for p in cap_emb.parameters():
            add(p, lr_head)

    return [dict(params=ps, lr=lr, weight_decay=w, base_lr=lr, is_enc=e)
            for (lr, w, e), ps in groups.items()]
