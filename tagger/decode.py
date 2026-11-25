"""Constrained decoding: factored-XPOS -> attested full tags; lemma via lexicon-rescored
edit scripts with an open-script fallback for OOV forms.

Morpheus hook: decode_lemma takes an optional candidate_fn(form_key, xpos) -> set[lemma]
that widens the in-vocab candidate set without any other code change.
"""
from __future__ import annotations

import numpy as np
import torch

from tagger.edits import XPOS_LEN, LabelVocab, apply_script, form_key


class TagDecoder:
    def __init__(self, vocab: LabelVocab, constrain_tags=True):
        self.vocab = vocab
        self.constrain = constrain_tags
        # (n_tags, 9) index matrix into per-position alphabets
        self.tag_idx = torch.tensor([vocab.xpos_ids(t) for t in vocab.tags], dtype=torch.long)
        assert int(self.tag_idx.min()) >= 0

    def xpos(self, xpos_logits, word_mask, flat_logits=None):
        """xpos_logits: list of 9 (B,W,|A_p|) tensors; flat_logits: optional (B,W,n_tags)
        full-tag head, combined additively -> list-of-lists of tag strings."""
        v = self.vocab
        if not self.constrain:
            out = []
            if flat_logits is not None:
                best = flat_logits.argmax(-1).cpu()
                return [[v.tags[int(best[b, w])]
                         for w in range(word_mask.shape[1]) if word_mask[b, w]]
                        for b in range(word_mask.shape[0])]
            preds = [lg.argmax(-1).cpu() for lg in xpos_logits]
            for b in range(word_mask.shape[0]):
                out.append(["".join(v.xpos_alpha[p][int(preds[p][b, w])]
                                    for p in range(XPOS_LEN))
                            for w in range(word_mask.shape[1]) if word_mask[b, w]])
            return out
        ti = self.tag_idx.to(xpos_logits[0].device)          # (n_tags, 9)
        score = 0
        for p, lg in enumerate(xpos_logits):
            lp = torch.log_softmax(lg.float(), -1)           # (B,W,|A_p|)
            score = score + lp[:, :, ti[:, p]]               # (B,W,n_tags)
        if flat_logits is not None:
            score = score + torch.log_softmax(flat_logits.float(), -1)
        best = score.argmax(-1).cpu()                        # (B,W)
        return [[v.tags[int(best[b, w])]
                 for w in range(word_mask.shape[1]) if word_mask[b, w]]
                for b in range(word_mask.shape[0])]

    def upos(self, upos_logits, word_mask):
        v = self.vocab
        pred = upos_logits.argmax(-1).cpu()
        return [[v.upos[int(pred[b, w])]
                 for w in range(word_mask.shape[1]) if word_mask[b, w]]
                for b in range(word_mask.shape[0])]


class LemmaDecoder:
    def __init__(self, vocab: LabelVocab, use_lexicon=True, candidate_fn=None, topk=64):
        self.vocab = vocab
        self.use_lexicon = use_lexicon
        self.candidate_fn = candidate_fn
        self.topk = topk
        # script applicability by form length: applicable iff p_cut + s_cut <= len(form)
        self._app_cache = {}
        self._pc = np.array([s[0] + s[2] for s in vocab.scripts])

    def _applicable(self, L):
        if L not in self._app_cache:
            self._app_cache[L] = torch.from_numpy(self._pc <= L)
        return self._app_cache[L]

    def __call__(self, form: str, script_logprobs: torch.Tensor, xpos: str | None = None) -> str:
        """script_logprobs: (n_scripts,) log-softmax for this word. xpos: predicted tag,
        used to prefer the (form, tag)-conditioned lexicon entry when attested."""
        v = self.vocab
        key = form_key(form)
        copy_cap = form[:1] != form[:1].lower()

        app = self._applicable(len(key)).to(script_logprobs.device)
        masked = script_logprobs.masked_fill(~app, -1e30)
        k = min(self.topk, masked.shape[-1])
        topv, topi = masked.topk(k)
        topv, topi = topv.tolist(), topi.tolist()

        if self.use_lexicon:
            cands = None
            if xpos is not None:
                cands = v.lex_ft.get(key + "\t" + xpos)
            if not cands:
                cands = v.lex_f.get(key, {})
            cands = dict(cands)
            if self.candidate_fn:
                for lem in self.candidate_fn(key, xpos) or ():
                    cands.setdefault(lem, 0)
            if cands:
                lower = {}
                for lemma in cands:
                    lower.setdefault(lemma.lower(), lemma)
                best, best_s = None, -1e30
                for s, i in zip(topv, topi):
                    if s <= -1e29:
                        break
                    out = apply_script(key, v.scripts[i])
                    lemma = lower.get(out)
                    if lemma is not None:
                        sc = s + 1e-3 * np.log1p(cands[lemma])   # attestation tiebreak
                        if sc > best_s:
                            best, best_s = lemma, sc
                # candidates outside the top-k script beam: fall back to attestation count
                return best if best is not None else max(cands, key=cands.get)

        # OOV path: best applicable script
        sc = v.scripts[topi[0]]
        lemma = apply_script(key, sc)
        if lemma is None:
            return form
        if sc[4] or copy_cap:
            lemma = lemma[:1].upper() + lemma[1:]
        return lemma
