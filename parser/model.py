"""Arm wrapper: frozen Stoicheia backbone + a learned scalar mix, producing per-sentence
word vectors for the biaffine head. Gold head/deprel alignment lives here too (tokens with no
Greek letters are skipped, exactly like the tagger pipeline skips them).

NOTE: this release drops the LemmaArm (a second arm over a LemmaDiff-grc encoder, fused or
compared against the char arm) and the "lemma"/"fused" SyntaxModel paths that depended on it —
LemmaDiff-grc is a separate, unpublished side-repo and out of scope here. The published joint
model (parser/joint_model.py, trained via parser/joint_train.py) supersedes those ablations
anyway: it beats both the char-only and lemma-only/fused specialists on test LAS. Only the char
arm below, and the "char"-only path in SyntaxModel/train.py/evaluate.py, remain."""
from __future__ import annotations

import torch
import torch.nn as nn

from tagger.dataset import encode_word
from parser.biaffine import ScalarMix, pool_words
from parser.pack import pack_char_rows, batch_char


def encodable_positions(sent):
    return [i for i, t in enumerate(sent.tokens) if encode_word(t.form) is not None]


def build_gold(sent, deprel_vocab):
    """-> (n, gold_head[n] (0=root, else 1..n), gold_label[n]) for the encodable tokens."""
    enc_pos = encodable_positions(sent)
    new_index = {p: i for i, p in enumerate(enc_pos)}
    n = len(enc_pos)
    heads = [-100] * n
    labels = [-100] * n
    for i, p in enumerate(enc_pos):
        t = sent.tokens[p]
        if not t.head.isdigit():
            continue
        h = int(t.head)
        if h == 0:
            heads[i] = 0
        else:
            hp = h - 1
            if hp in new_index:
                heads[i] = new_index[hp] + 1
            else:
                continue          # gold head was a skipped (non-Greek) token: exclude
        labels[i] = deprel_vocab.id(t.deprel)
    return n, heads, labels


def gold_tensors(sents, deprel_vocab, device):
    infos = [build_gold(s, deprel_vocab) for s in sents]
    maxW = max((n for n, _, _ in infos), default=0)
    B = len(sents)
    heads = torch.full((B, maxW), -100, dtype=torch.long)
    labels = torch.full((B, maxW), -100, dtype=torch.long)
    mask = torch.zeros((B, maxW), dtype=torch.bool)
    for b, (n, h, l) in enumerate(infos):
        if n == 0:
            continue
        heads[b, :n] = torch.tensor(h)
        labels[b, :n] = torch.tensor(l)
        mask[b, :n] = True
    return heads.to(device), labels.to(device), mask.to(device), maxW


def _scatter(pooled, slots, pairs, n_sents, max_words):
    """Place each word vector at its COMPACTED encodable index (0,1,2,… in word order),
    matching build_gold's indexing. NOTE: do NOT use the raw token index `ti` here — the
    char arm's `ti` is the position in sent.tokens (with skipped non-Greek tokens leaving
    gaps), which shifts every vector off its gold head. A per-sentence running counter over
    the words in emitted order is the encodable index for both arms (a no-op for the lemma
    arm, whose `ti` is already compacted)."""
    D = pooled.shape[-1]
    out = pooled.new_zeros(n_sents, max_words, D)
    counter = {}
    for r, rs in enumerate(slots):
        for w, (si, ti) in enumerate(rs):
            orig = pairs[si][0]
            idx = counter.get(orig, 0)
            counter[orig] = idx + 1
            if idx < max_words:
                out[orig, idx] = pooled[r, w]
    return out


class CharArm(nn.Module):
    def __init__(self, char_model, n_layers, finetune=False):
        super().__init__()
        self.model = char_model                       # CharBertWithHidden
        self.finetune = finetune
        if not finetune:                              # frozen probe: no backbone grads
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.model.return_layers = True
        self.mix = ScalarMix(n_layers)

    def out_dim(self):
        return self.model.cfg.d_model

    def _encode_rows(self, rows, T, W, device):
        b, slots = batch_char(rows, T, W, device)
        ctx = torch.enable_grad() if self.finetune else torch.no_grad()
        with ctx:
            out = self.model(b)
        return out, b, slots

    def forward(self, sents, T, W, device, micro=8, max_words=None):
        rows, pairs, _ = pack_char_rows(sents, T, W)
        pooled_all, slots_all, rowbase = [], [], 0
        for r0 in range(0, len(rows), micro):
            chunk = rows[r0:r0 + micro]
            out, b, slots = self._encode_rows(chunk, T, W, device)
            layers = out["layers"] + [out["hidden"]]   # 32 blocks + final norm
            mixed = self.mix(layers)                    # grad flows (mix + backbone if finetune)
            pooled = pool_words(mixed, b["word_id"], W, "mean")
            pooled_all.append(pooled)
            slots_all.extend(slots)
        pooled = torch.cat(pooled_all, 0) if pooled_all else \
            torch.zeros(0, W, self.out_dim(), device=device)
        mw = max_words if max_words is not None else W
        return _scatter(pooled, slots_all, pairs, len(sents), mw)


# LemmaArm class removed for this release (depended on the unpublished LemmaDiff-grc
# encoder + ldf.model.lemma_diff — see module docstring above).


class SyntaxModel(nn.Module):
    """Only arm="char" is supported in this release (see module docstring)."""
    def __init__(self, arm, char_arm, lemma_arm, head):
        super().__init__()
        assert arm == "char", 'only arm="char" is supported in this release (lemma/fused dropped)'
        self.arm = arm
        self.char_arm = char_arm
        self.lemma_arm = lemma_arm     # always None in this release; kept for state_dict shape parity
        self.head = head

    def word_vectors(self, sents, T, W, device, max_words):
        return self.char_arm(sents, T, W, device, max_words=max_words)

    def forward(self, sents, deprel_vocab, T, W, device):
        heads, labels, mask, maxW = gold_tensors(sents, deprel_vocab, device)
        if maxW == 0:
            return None, None, None, None, mask
        w = self.word_vectors(sents, T, W, device, maxW)
        arc_scores, rel_scores = self.head(w, mask)
        return arc_scores, rel_scores, heads, labels, mask
