"""Tag raw polytonic Greek text end-to-end.

  python -m tagger.tag --run $STOICHEIA_DATA/runs/tagger_fold0_pilot --text "..." [--tsv out.tsv]
  echo "..." | python -m tagger.tag --run ...

Tokenization is the pretraining normalizer's (whitespace/punctuation): crasis and
elision are NOT split into multiple syntactic words the way AGDT does, so such tokens
get a single best-effort analysis.
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tagger.backbone import Stats, normalize_record, restore_polytonic
from tagger.dataset import SentEnc, batch_rows, pack_rows
from tagger.decode import LemmaDecoder, TagDecoder
from tagger.evaluate import load_run


def encode_raw(text):
    """-> (sent_encs, forms_per_sentence). All planes carry true raw-text values."""
    r = normalize_record(text, Stats(), with_punct=True)
    if r is None:
        return [], []
    chars, boundary, dia, cap, punct = r
    words = restore_polytonic(chars, dia, cap, boundary)
    ends = np.flatnonzero(boundary >= 1)
    sents, forms = [], []
    s0, w0 = 0, 0
    for k, e in enumerate(ends):
        if boundary[e] == 2 or k == len(ends) - 1:
            sl = slice(s0, e + 1)
            wends = ends[w0:k + 1] - s0
            spans, prev = [], 0
            for we in wends:
                spans.append((prev, int(we) + 1))
                prev = int(we) + 1
            n = len(spans)
            sents.append(SentEnc(chars=chars[sl], boundary=boundary[sl], dia=dia[sl],
                                 punct=punct[sl], cap=cap[sl], spans=spans,
                                 y_xpos=np.full((n, 9), -100, dtype=np.int64),
                                 y_script=np.full(n, -100, dtype=np.int64),
                                 y_upos=np.full(n, -100, dtype=np.int64),
                                 y_tag=np.full(n, -100, dtype=np.int64)))
            forms.append(words[w0:k + 1])
            s0, w0 = e + 1, k + 1
    return sents, forms


@torch.no_grad()
def tag_text(model, vocab, text, device, T, W, micro=16):
    sents, forms = encode_raw(text)
    if not sents:
        return []
    tagd, lemd = TagDecoder(vocab), LemmaDecoder(vocab)
    out_rows = [[None] * len(f) for f in forms]
    rows, _ = pack_rows(sents, T, W)
    # word slots in raw mode are indexed by span order == token order
    slot_of = []
    for row in rows:
        rs = []
        for si, e in row.sents:
            rs.extend((si, ti) for ti in range(len(e.spans)))
        slot_of.append(rs)
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        batch = batch_rows(chunk, T, W)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b)
        mask = batch["word_id"].new_zeros(len(chunk), W, dtype=torch.bool)
        for bi, rs in enumerate(slot_of[i:i + micro]):
            mask[bi, :len(rs)] = True
        xp = tagd.xpos(out["xpos"], mask, out.get("flat"))
        up = tagd.upos(out["upos"], mask)
        slp = torch.log_softmax(out["script"].float(), -1)
        for bi, rs in enumerate(slot_of[i:i + micro]):
            lp = slp[bi].cpu()
            for w, (si, ti) in enumerate(rs):
                form = forms[si][ti]
                out_rows[si][ti] = (form, lemd(form, lp[w], xpos=xp[bi][w]),
                                    up[bi][w], xp[bi][w])
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--text", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, sd = load_run(os.path.expandvars(a.run), device)
    text = a.text if a.text is not None else sys.stdin.read()
    for sent in tag_text(model, vocab, text, device, sd["T"], sd["W"]):
        for form, lemma, upos, xpos in sent:
            print(f"{form}\t{lemma}\t{upos}\t{xpos}")
        print()


if __name__ == "__main__":
    main()
