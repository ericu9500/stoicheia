"""HF-Hub-compatible model for Stoicheia-tagger-parser (JointModel: tagger + biaffine parser).

Self-contained: vendors the transformer primitives (RMSNorm/RoPE/Attention/GeGLU/Block/
build_attn_mask) verbatim from modeling_char_bert.py, plus the tagger/parser primitives
(pool_words, MLP, Biaffine) verbatim from tagger/model.py and parser/biaffine.py. SDPA-only
attention path (same portability tradeoff as the base CharBertModel wrapper).

Architecture mirrors, exactly, the original training-time module tree so a real checkpoint's
state dict loads with strict=True (this is the acid test the conversion script relies on):

    tagger.encoder.*            CharBertEncoder (+ a fine-tune-only `cap_emb` channel)
    tagger.mix_w                 (depth+1,) ELMo-style scalar-mix weights over layer outputs
    tagger.xpos_heads.{0..8}     9 factored XPOS position heads
    tagger.head_flat             flat full-XPOS-tag head (attested tags only)
    tagger.head_script           lemma edit-script head
    tagger.head_upos             UPOS head
    biaffine.root                learnable ROOT vector (head-candidate column 0)
    biaffine.{arc,rel}_{dep,head}  MLPs projecting pooled word vectors for arc/label scoring
    biaffine.{arc,rel}_biaf      bilinear (Dozat & Manning) scorers

One notable, documented simplification vs. the original training code: the original packs
several sentences into shared, fixed-length rows for compute efficiency during training/eval
(tagger/dataset.py's pack_rows + parser/joint_model.py's JointModel._regroup step, which
un-packs pooled word vectors back into one (n_sent, max_w, D) tensor per sentence before the
biaffine head). For a standalone inference wrapper there is no efficiency reason to pack
multiple sentences per row, so this module processes exactly one sentence per batch row --
this makes the `_regroup` step an identity (sent_ids = range(B), no gather needed) while
remaining architecturally identical: forward() still pools per-character hidden states into
per-word vectors via a `word_id` tensor, exactly as during training, before every head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_char_bert_joint import CharBertJointConfig

# ============================================================================================
# Vendored CharBertEncoder primitives (verbatim from modeling_char_bert.py)
# ============================================================================================


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


class RoPE(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base

    def cos_sin(self, pos):
        # Recomputed on every call rather than cached in a registered buffer: a
        # persistent=False buffer is never covered by the checkpoint's state dict,
        # so it depends entirely on __init__-time materialization -- which some
        # transformers versions' meta-device/low_cpu_mem_usage loading path can
        # skip, silently leaving this tensor uninitialized. Recomputing here is
        # immune to that regardless of how the model was constructed/loaded.
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=pos.device).float() / self.dim))
        f = torch.outer(pos.float(), inv)
        emb = torch.cat([f, f], -1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], -1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None]
    sin = sin[None, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, d, n_heads, rope: RoPE, qk_norm=False):
        super().__init__()
        self.h = n_heads
        self.dh = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = rope
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.dh)
            self.k_norm = RMSNorm(self.dh)

    def forward(self, x, pos, attn_mask):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = self.rope.cos_sin(pos)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o(out)


class GeGLU(nn.Module):
    def __init__(self, d, mult=8 / 3):
        super().__init__()
        hidden = int(d * mult)
        hidden = (hidden + 63) // 64 * 64
        self.wi = nn.Linear(d, 2 * hidden, bias=False)
        self.wo = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, -1)
        return self.wo(F.gelu(a) * b)


class Block(nn.Module):
    def __init__(self, d, n_heads, rope, window=0, qk_norm=False):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = Attention(d, n_heads, rope, qk_norm=qk_norm)
        self.n2 = RMSNorm(d)
        self.mlp = GeGLU(d)
        self.window = window  # 0 = global; >0 = local sliding window (characters)

    def forward(self, x, pos, base_mask):
        x = x + self.attn(self.n1(x), pos, base_mask)
        x = x + self.mlp(self.n2(x))
        return x


def build_attn_mask(seg_id, window, device, dtype):
    """Additive mask (B,1,T,T): same-segment AND (window==0 or |i-j|<window)."""
    B, T = seg_id.shape
    same = seg_id[:, None, :] == seg_id[:, :, None]
    if window and window > 0:
        idx = torch.arange(T, device=device)
        near = (idx[None, :] - idx[:, None]).abs() < window
        same = same & near[None]
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    mask.masked_fill_(~same[:, None], float("-inf"))
    return mask


# ============================================================================================
# Vendored tagger/parser primitives (verbatim from tagger/model.py and parser/biaffine.py)
# ============================================================================================


def pool_words(hidden, word_id, W, mode="mean"):
    """hidden (B,T,D), word_id (B,T) in [-1,W) -> (B,W,D)."""
    B, T, D = hidden.shape
    flat = hidden.reshape(B * T, D)
    wid = word_id.reshape(B * T)
    valid = wid >= 0
    off = (torch.arange(B, device=hidden.device) * W).repeat_interleave(T)
    idx = (wid + off)[valid]
    out = hidden.new_zeros(B * W, D)
    if mode == "mean":
        out.index_add_(0, idx, flat[valid])
        cnt = hidden.new_zeros(B * W).index_add_(
            0, idx, torch.ones_like(idx, dtype=hidden.dtype))
        out = out / cnt.clamp(min=1).unsqueeze(-1)
    elif mode == "last":
        out.index_copy_(0, idx, flat[valid])   # spans are contiguous: last write = last char
    else:
        raise ValueError(mode)
    return out.reshape(B, W, D)


class MLP(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.33):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.lin(x)))


class Biaffine(nn.Module):
    """s(x,y) = [x;1]^T W [y;1] (per output channel). x:(B,Lx,Di) y:(B,Ly,Di) -> (B,n_out,Lx,Ly)."""
    def __init__(self, d_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x, self.bias_y = bias_x, bias_y
        self.W = nn.Parameter(torch.zeros(n_out, d_in + int(bias_x), d_in + int(bias_y)))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], -1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], -1)
        s = torch.einsum("bxi,oij,byj->boxy", x, self.W, y)
        return s.squeeze(1) if s.shape[1] == 1 else s


# ============================================================================================
# Encoder (CharBertEncoder + fine-tune-only cap_emb), lives at `model.tagger.encoder.*`
# ============================================================================================


class _Encoder(nn.Module):
    def __init__(self, config: CharBertJointConfig):
        super().__init__()
        self.e_char = nn.Embedding(config.n_char_ids, config.d_model)
        self.e_bnd = nn.Embedding(config.n_boundary, config.d_model)
        self.e_dia = nn.Embedding(config.n_dia, config.d_model)
        self.e_punct = nn.Embedding(config.n_punct, config.d_model)
        rope = RoPE(config.d_model // config.n_heads)
        blocks = []
        for i in range(config.depth):
            win = 0 if i % 4 == 3 else config.char_window  # 3 local : 1 global
            blocks.append(Block(config.d_model, config.n_heads, rope, window=win, qk_norm=config.qk_norm))
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = RMSNorm(config.d_model)
        self.head_char = nn.Linear(config.d_model, config.n_char_ids, bias=False)
        self.head_bnd = nn.Linear(config.d_model, 3, bias=False)
        self.head_dia = nn.Linear(config.d_model, 48, bias=False)
        self.head_cap = nn.Linear(config.d_model, 2, bias=False)
        self.head_punct = nn.Linear(config.d_model, 6, bias=False)
        if config.use_cap:
            # fine-tune-only additive capitalization channel (pretraining treats cap as
            # output-only); zero-init so loading a pretraining checkpoint would be a no-op.
            self.cap_emb = nn.Embedding(2, config.d_model)

    def forward(self, input_ids, boundary, dia, punct, cap=None, seg_id=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        seg = seg_id if seg_id is not None else torch.zeros(B, T, dtype=torch.long, device=input_ids.device)

        x = self.e_char(input_ids) + self.e_bnd(boundary) + self.e_dia(dia) + self.e_punct(punct)
        cap_emb = getattr(self, "cap_emb", None)
        if cap_emb is not None and cap is not None:
            x = x + cap_emb(cap)

        char_window = None
        for blk in self.blocks:
            if blk.window == 0:
                continue
            char_window = blk.window
            break
        if char_window is None:
            char_window = 0
        attn_mask = build_attn_mask(seg, char_window, input_ids.device, x.dtype)
        glob_mask = build_attn_mask(seg, 0, input_ids.device, x.dtype)

        layers = []
        for blk in self.blocks:
            m = glob_mask if blk.window == 0 else attn_mask
            x = blk(x, pos, m)
            layers.append(x)

        hidden = self.norm_out(x)
        return dict(
            layers=tuple(layers),
            hidden=hidden,
            char=self.head_char(hidden),
            boundary=self.head_bnd(hidden),
            dia=self.head_dia(hidden),
            cap=self.head_cap(hidden),
            punct=self.head_punct(hidden),
        )


# ============================================================================================
# Tagger head bundle: lives at `model.tagger.*`
# ============================================================================================


class _Tagger(nn.Module):
    def __init__(self, config: CharBertJointConfig):
        super().__init__()
        self.encoder = _Encoder(config)
        self.mix_w = nn.Parameter(torch.zeros(config.depth + 1))
        self.dropout = nn.Dropout(config.head_dropout)
        self.xpos_heads = nn.ModuleList(
            [nn.Linear(config.d_model, n, bias=False) for n in config.n_xpos_classes])
        self.head_flat = (nn.Linear(config.d_model, config.n_flat_tags, bias=False)
                           if config.use_flat else None)
        self.head_script = nn.Linear(config.d_model, config.n_script, bias=False)
        self.head_upos = nn.Linear(config.d_model, config.n_upos, bias=False)


# ============================================================================================
# Biaffine parser head: lives at `model.biaffine.*`
# ============================================================================================


class _BiaffineHead(nn.Module):
    def __init__(self, config: CharBertJointConfig):
        super().__init__()
        d = config.d_model
        self.root = nn.Parameter(torch.zeros(d))
        self.arc_dep = MLP(d, config.d_arc, config.parse_dropout)
        self.arc_head = MLP(d, config.d_arc, config.parse_dropout)
        self.rel_dep = MLP(d, config.d_rel, config.parse_dropout)
        self.rel_head = MLP(d, config.d_rel, config.parse_dropout)
        self.arc_biaf = Biaffine(config.d_arc, n_out=1, bias_x=True, bias_y=False)
        self.rel_biaf = Biaffine(config.d_rel, n_out=config.n_labels, bias_x=True, bias_y=True)

    def forward(self, w, word_mask):
        """w: (B,W,D) word vectors. word_mask: (B,W) bool, True at real words.
        Returns arc_scores (B,W,W+1) [col0=root], rel_scores (B,W,W+1,n_labels)."""
        B, W, D = w.shape
        root = self.root.view(1, 1, D).expand(B, 1, D)
        heads_in = torch.cat([root, w], 1)                       # (B,W+1,D): col0=root
        h_dep_arc = self.arc_dep(w)                               # (B,W,d_arc)
        h_head_arc = self.arc_head(heads_in)                      # (B,W+1,d_arc)
        arc_scores = self.arc_biaf(h_dep_arc, h_head_arc)          # (B,W,W+1)
        # mask: dependent i cannot pick itself as head (col i+1), and padded cols get -inf
        pad_head = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=w.device), word_mask], 1)
        arc_scores = arc_scores.masked_fill(~pad_head[:, None, :], float("-inf"))
        self_idx = torch.arange(W, device=w.device)
        arc_scores = arc_scores.clone()
        arc_scores[:, self_idx, self_idx + 1] = float("-inf")

        h_dep_rel = self.rel_dep(w)                                # (B,W,d_rel)
        h_head_rel = self.rel_head(heads_in)                       # (B,W+1,d_rel)
        rel_scores = self.rel_biaf(h_dep_rel, h_head_rel)           # (B,n_labels,W,W+1)
        rel_scores = rel_scores.permute(0, 2, 3, 1)                 # (B,W,W+1,n_labels)
        return arc_scores, rel_scores

    @torch.no_grad()
    def decode(self, arc_scores, rel_scores):
        """Greedy per-token argmax head (col 0=root) + label argmax at the chosen head."""
        heads_out = arc_scores.argmax(-1)                          # (B,W) in [0..W], 0=root
        B, W = heads_out.shape
        bi = torch.arange(B, device=arc_scores.device)[:, None].expand(B, W)
        wi = torch.arange(W, device=arc_scores.device)[None, :].expand(B, W)
        labels_out = rel_scores[bi, wi, heads_out].argmax(-1)
        return heads_out.cpu(), labels_out.cpu()


# ============================================================================================
# Output dataclass
# ============================================================================================


@dataclass
class CharBertJointOutput(ModelOutput):
    xpos_logits: Optional[tuple] = None       # tuple of 9 (B,W,|A_p|) tensors
    script_logits: torch.FloatTensor = None   # (B,W,n_script)
    upos_logits: torch.FloatTensor = None     # (B,W,n_upos)
    flat_logits: Optional[torch.FloatTensor] = None   # (B,W,n_flat_tags)
    arc_scores: Optional[torch.FloatTensor] = None    # (B,W,W+1), col0 = root
    rel_scores: Optional[torch.FloatTensor] = None    # (B,W,W+1,n_labels)
    word_mask: Optional[torch.BoolTensor] = None       # (B,W)
    hidden_states: Optional[tuple] = None


# ============================================================================================
# Top-level model
# ============================================================================================


class CharBertForTaggingAndParsing(PreTrainedModel):
    config_class = CharBertJointConfig

    def __init__(self, config: CharBertJointConfig):
        super().__init__(config)
        self.tagger = _Tagger(config)
        self.biaffine = _BiaffineHead(config)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        boundary: torch.LongTensor,
        dia: torch.LongTensor,
        punct: torch.LongTensor,
        word_id: torch.LongTensor,
        cap: Optional[torch.LongTensor] = None,
        seg_id: Optional[torch.LongTensor] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        cfg = self.config
        enc_out = self.tagger.encoder(input_ids, boundary, dia, punct, cap=cap, seg_id=seg_id)
        layers = (*enc_out["layers"], enc_out["hidden"])   # depth raw block outs + 1 normed final

        B, T = input_ids.shape
        device = input_ids.device
        W = int(word_id.max().item()) + 1 if word_id.numel() and bool((word_id >= 0).any()) else 0
        if W == 0:
            # degenerate: no encodable words anywhere in the batch
            w = input_ids.new_zeros(B, 0, cfg.d_model, dtype=layers[0].dtype)
            word_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
        else:
            pooled = torch.stack(
                [pool_words(h, word_id, W, cfg.pool) for h in layers])         # (L+1,B,W,D)
            mix = torch.softmax(self.tagger.mix_w, 0)
            w = torch.einsum("l,lbwd->bwd", mix.to(pooled.dtype), pooled)      # (B,W,D), no dropout
            valid = word_id >= 0
            cnt = torch.zeros(B, W, device=device, dtype=torch.float32)
            if bool(valid.any()):
                idx_b = torch.arange(B, device=device).unsqueeze(1).expand(B, T)[valid]
                idx_w = word_id[valid]
                cnt.index_put_((idx_b, idx_w), torch.ones_like(idx_w, dtype=torch.float32), accumulate=True)
            word_mask = cnt > 0

        # tag heads see a dropped-out copy (a no-op in eval mode); the biaffine head sees the
        # raw pooled `w`, exactly as parser.joint_model.JointModel separates the two (dropout is
        # applied inside TaggerModel._tag_heads on a *local* copy, never touching the tensor
        # that JointModel._regroup / BiaffineHead consume).
        w_drop = self.tagger.dropout(w)
        xpos_logits = tuple(hd(w_drop) for hd in self.tagger.xpos_heads)
        script_logits = self.tagger.head_script(w_drop)
        upos_logits = self.tagger.head_upos(w_drop)
        flat_logits = self.tagger.head_flat(w_drop) if self.tagger.head_flat is not None else None

        if W > 0:
            arc_scores, rel_scores = self.biaffine(w, word_mask)
        else:
            arc_scores, rel_scores = None, None

        if not return_dict:
            return (xpos_logits, script_logits, upos_logits, flat_logits, arc_scores, rel_scores, word_mask)
        return CharBertJointOutput(
            xpos_logits=xpos_logits,
            script_logits=script_logits,
            upos_logits=upos_logits,
            flat_logits=flat_logits,
            arc_scores=arc_scores,
            rel_scores=rel_scores,
            word_mask=word_mask,
            hidden_states=layers,
        )
