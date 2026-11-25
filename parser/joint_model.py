"""JointModel: one fine-tuned Stoicheia backbone -> word vectors -> ALL CoNLL-U columns.

Shares the tagger's pooled + scalar-mixed word representation between:
  * the tagger heads (factored XPOS / lemma edit-script / UPOS / flat-tag) — row-level, and
  * a Dozat-Manning biaffine head (HEAD + DEPREL) — regrouped per sentence.

The tagger packs several sentences per row (block-diagonal attention) and never splits a
sentence across rows, so each word slot's (sentence_index, token_index) in `batch['slots']`
lets us gather a sentence's word vectors back into a contiguous (n_sent, max_w, D) tensor for
the arc/label scorer. Word order within a sentence is token order over *encodable* tokens —
exactly parser.model.build_gold's indexing, so gold heads/labels line up slot-for-slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tagger.model import TaggerModel, pool_words
from parser.biaffine import BiaffineHead


class JointModel(nn.Module):
    def __init__(self, encoder, vocab, tcfg, pcfg, W=384):
        super().__init__()
        self.tagger = TaggerModel(encoder, vocab, tcfg, W=W)   # heads + scalar mix + LM-head freeze
        self.biaffine = BiaffineHead(encoder.cfg.d_model, pcfg)
        self.W = W

    # ---- shared word representation (mirrors TaggerModel.forward's pooling) --------------
    def _word_vectors(self, batch):
        enc = self.tagger.encoder
        out = enc(batch)
        if self.tagger.tcfg.scalar_mix:
            pooled = torch.stack(
                [pool_words(h, batch["word_id"], self.W, self.tagger.tcfg.pool)
                 for h in [*out["layers"], out["hidden"]]])         # (L+1,B,W,D)
            mix = torch.softmax(self.tagger.mix_w, 0)
            w = torch.einsum("l,lbwd->bwd", mix.to(pooled.dtype), pooled)
        else:
            w = pool_words(out["hidden"], batch["word_id"], self.W, self.tagger.tcfg.pool)
        return w

    def _tag_heads(self, w):
        tg = self.tagger
        wt = tg.dropout(w)
        r = dict(xpos=[hd(wt) for hd in tg.xpos_heads],
                 script=tg.head_script(wt),
                 upos=tg.head_upos(wt))
        if tg.head_flat is not None:
            r["flat"] = tg.head_flat(wt)
        return r

    @staticmethod
    def _regroup(w, slots):
        """(B,W,D) row-packed word vectors -> (n_sent, max_w, D) per-sentence, using slots.
        Returns (w_sent, word_mask, sent_ids) where sent_ids[local] is the global sentence
        index and word position = running per-sentence counter (== build_gold order)."""
        B, W, D = w.shape
        sent_ids, sid_to_local, pos_counter = [], {}, {}
        b_idx, k_idx, l_idx, p_idx = [], [], [], []
        for b, rs in enumerate(slots):
            for k, (si, _ti) in enumerate(rs):
                if si not in sid_to_local:
                    sid_to_local[si] = len(sent_ids); sent_ids.append(si)
                local = sid_to_local[si]
                pos = pos_counter.get(local, 0); pos_counter[local] = pos + 1
                b_idx.append(b); k_idx.append(k); l_idx.append(local); p_idx.append(pos)
        n_sent = len(sent_ids)
        max_w = max(pos_counter.values(), default=0)
        w_sent = w.new_zeros(n_sent, max_w, D)
        mask = torch.zeros(n_sent, max_w, dtype=torch.bool, device=w.device)
        if b_idx:
            dev = w.device
            bs = torch.tensor(b_idx, device=dev); ks = torch.tensor(k_idx, device=dev)
            ls = torch.tensor(l_idx, device=dev); ps = torch.tensor(p_idx, device=dev)
            w_sent[ls, ps] = w[bs, ks]           # differentiable advanced-index gather
            mask[ls, ps] = True
        return w_sent, mask, sent_ids

    def forward(self, batch):
        """-> (tag_out, arc_scores, rel_scores, word_mask, sent_ids). Loss/gold in the trainer
        (it holds the Sentence objects); all trainable submodules run inside this one forward
        so DDP sees the whole graph each step."""
        w = self._word_vectors(batch)
        tag_out = self._tag_heads(w)
        w_sent, word_mask, sent_ids = self._regroup(w, batch["slots"])
        if w_sent.shape[1] == 0:
            return tag_out, None, None, word_mask, sent_ids
        arc_scores, rel_scores = self.biaffine(w_sent, word_mask)
        return tag_out, arc_scores, rel_scores, word_mask, sent_ids
