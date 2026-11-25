"""Ithaca-protocol restoration eval for the flat Stoicheia/Stoicheia torso.

Beam-20 non-sequential iterative mask-predict (port of grc-encoder's faithful Ithaca
beam): mask one contiguous span of length L in a test/val segment; each round, forward
every hypothesis, rank all (masked position, letter) pairs, commit the best per child,
repeat until the gap is full. Metrics per L and averaged over L=1..10 (Ithaca reports
CER 26.3%, top-1 61.8%, top-20 78.3% on their protocol; our CER is letters-only —
word boundaries live in a separate channel — noted as a protocol delta).

  python insc_eval/restore.py --ckpt $INS_TORSO --split val --n 200
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from data.normalize import ALPHABET
from eval.intrinsic import load_model
from iphi import load as load_iphi
from papyri import load as load_papyri
from meta_vocab import UNK_REGION, UNK_CENTURY

ALIST = list(ALPHABET)
MASK, NLET = 24, 24
UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6


def levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


@torch.no_grad()
def _char_logp(model, seqs, bnd_row, device, region_id=UNK_REGION, century_id=UNK_CENTURY):
    """seqs: list of same-length int arrays. Boundary known OUTSIDE the gap (stone
    preserves word dividers around a lacuna); dia/punct unknown. region_id/century_id are
    per-inscription (constant across the row) -- UNK for a non-metadata-conditioned model,
    which ignores them regardless. -> (B,T,24) log-probs."""
    B, T = len(seqs), len(seqs[0])
    ids = torch.tensor(np.stack(seqs), dtype=torch.long, device=device)
    bnd = torch.tensor(bnd_row, dtype=torch.long, device=device)[None].expand(B, T).contiguous()
    batch = dict(input_ids=ids, boundary=bnd,
                 dia=torch.full((B, T), UNK_DIA, dtype=torch.long, device=device),
                 punct=torch.full((B, T), UNK_PUNCT, dtype=torch.long, device=device),
                 region=torch.full((B, T), region_id, dtype=torch.long, device=device),
                 century=torch.full((B, T), century_id, dtype=torch.long, device=device),
                 seg_id=torch.ones(B, T, dtype=torch.long, device=device))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch)
    return torch.log_softmax(out["char"][:, :, :NLET].float(), -1)


@torch.no_grad()
def beam_restore(model, chars, gap, bnd_row, device, beam_width=20, expand=48,
                  region_id=UNK_REGION, century_id=UNK_CENTURY):
    base = np.asarray(chars, dtype=np.int64)
    beam = [(base.copy(), tuple(gap), 0.0)]
    finished = {}
    while beam:
        logp = _char_logp(model, [h[0] for h in beam], bnd_row, device, region_id, century_id)
        children = {}
        for bi, (seq, rem, score) in enumerate(beam):
            rem = list(rem)
            sub = logp[bi, rem]                          # (len(rem), 24)
            flat = sub.reshape(-1)
            top = torch.topk(flat, min(expand, flat.numel())).indices.cpu().numpy()
            for t in top:
                pi, ch = divmod(int(t), NLET)
                pos = rem[pi]
                new = seq.copy(); new[pos] = ch
                nrem = tuple(p for p in rem if p != pos)
                ns = score + float(sub[pi, ch])
                if not nrem:
                    key = new[gap].tobytes()
                    if key not in finished or finished[key][1] < ns:
                        finished[key] = ("".join(ALIST[c] for c in new[gap]), ns)
                else:
                    key = (new.tobytes(), nrem)
                    if key not in children or children[key][2] < ns:
                        children[key] = (new, nrem, ns)
        beam = sorted(children.values(), key=lambda h: -h[2])[:beam_width]
    return sorted(finished.values(), key=lambda x: -x[1])[:beam_width]


def eval_span(model, recs, L, device, n, beam_width=20, seed=0, ctx=768, force_unk_metadata=False,
             verbose=False):
    """force_unk_metadata=True ignores each record's own region_id/century_id and scores
    with both forced to UNK regardless -- the "metadata withheld" condition, used both for
    the with-vs-without ablation and for the Ithaca-comparable run (Ithaca has no metadata-
    conditioning capability at all, so that comparison must not give this model an input
    Ithaca structurally can't have)."""
    rng = np.random.default_rng(seed + L)
    cers, t1, t20, tot = [], 0, 0, 0
    for r in recs[:]:
        chars = np.asarray(r["chars"], np.int64)
        if len(chars) <= L + 8:
            continue
        s = int(rng.integers(4, len(chars) - L - 4))
        lo = max(0, s - ctx // 2); hi = min(len(chars), s + L + ctx // 2)
        window = chars[lo:hi].copy()
        gap = list(range(s - lo, s - lo + L))
        gold = "".join(ALIST[c] for c in window[gap])
        window[gap] = MASK
        bnd = np.minimum(np.asarray(r["boundary"][lo:hi]), 2).astype(np.int64)
        bnd[gap] = UNK_BND                     # boundary unknown INSIDE the gap
        region_id = UNK_REGION if force_unk_metadata else r.get("region_id", UNK_REGION)
        century_id = UNK_CENTURY if force_unk_metadata else r.get("century_id", UNK_CENTURY)
        cand = beam_restore(model, window, gap, bnd, device, beam_width,
                            region_id=region_id, century_id=century_id)
        if not cand:
            continue
        pred = cand[0][0]
        cer = levenshtein(pred, gold) / max(len(gold), 1)
        cers.append(cer)
        hit1 = int(pred == gold); hit20 = int(any(c[0] == gold for c in cand))
        t1 += hit1; t20 += hit20
        tot += 1
        if verbose:
            print(f"    [L={L} {tot}/{n}] gold={gold!r:>12} pred={pred!r:>12} "
                  f"cer={cer:.2f} top1={hit1} top20={hit20}", flush=True)
        if tot >= n:
            break
    return dict(L=L, n=tot, CER=round(float(np.mean(cers)), 4),
                top1=round(t1 / max(tot, 1), 4), top20=round(t20 / max(tot, 1), 4))


def eval_span_whole(model, recs, L, device, n, beam_width=20, seed=0, T_char=4096,
                    force_unk_metadata=False, verbose=False):
    """The realistic task, as actually stated: take an inscription/papyrus AS EDITED -- the
    whole document, real lacunae intact exactly where the edition has them -- and fill in
    ONE blank. NOT a cropped snippet between two lacunae: the entire document is the context
    (only capped by T_char in the rare very-long-document tail, same fallback used
    everywhere else in this design), so a real lacuna elsewhere in the SAME document sits in
    view, unresolved, exactly as it would at actual deployment. This is what the standard
    eval_span() (clean split-at-lacuna segments, no real gap ever in context) cannot test."""
    rng = np.random.default_rng(seed + L)
    cers, t1, t20, tot = [], 0, 0, 0
    for r in recs[:]:
        chars = np.asarray(r["chars"], np.int64)
        real_lac = np.asarray(r["is_real_lacuna"], dtype=bool)
        if len(chars) <= L + 8 or len(chars) > T_char:
            continue
        # candidate start positions: an L-wide run entirely within KNOWN text (never overlap
        # a real lacuna -- that would have no gold answer to score against)
        knownable = ~real_lac
        valid_starts = [s for s in range(4, len(chars) - L - 4)
                        if knownable[s:s + L].all()]
        if not valid_starts:
            continue
        s = int(rng.choice(valid_starts))
        window = chars.copy()
        gap = list(range(s, s + L))
        gold = "".join(ALIST[c] for c in window[gap])
        window[gap] = MASK
        bnd = np.minimum(np.asarray(r["boundary"]), 2).astype(np.int64)
        bnd[gap] = UNK_BND                     # boundary unknown INSIDE the synthetic gap
        # real lacunae elsewhere in the SAME document are untouched: already MASK/UNK_BND
        # from load_whole_full()'s own encoding -- exactly what the model trained on
        region_id = UNK_REGION if force_unk_metadata else r.get("region_id", UNK_REGION)
        century_id = UNK_CENTURY if force_unk_metadata else r.get("century_id", UNK_CENTURY)
        cand = beam_restore(model, window, gap, bnd, device, beam_width,
                            region_id=region_id, century_id=century_id)
        if not cand:
            continue
        pred = cand[0][0]
        cer = levenshtein(pred, gold) / max(len(gold), 1)
        cers.append(cer)
        hit1 = int(pred == gold)
        hit20 = int(any(c[0] == gold for c in cand))
        t1 += hit1
        t20 += hit20
        tot += 1
        if verbose:
            n_lac = int(real_lac.sum())
            print(f"    [L={L} {tot}/{n}] gold={gold!r:>12} pred={pred!r:>12} "
                  f"cer={cer:.2f} top1={hit1} top20={hit20} doc_len={len(chars)} "
                  f"real_lacuna_chars_elsewhere={n_lac}", flush=True)
        if tot >= n:
            break
    return dict(L=L, n=tot, CER=round(float(np.mean(cers)), 4),
                top1=round(t1 / max(tot, 1), 4), top20=round(t20 / max(tot, 1), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--domain", default="iphi", choices=["iphi", "papyri"],
                    help="iphi = inscriptions (region/tpq/taq available); papyri = "
                         "documentary papyri (no date/place metadata in this corpus, "
                         "always UNK region/century)")
    ap.add_argument("--mode", default="clean", choices=["clean", "whole"],
                    help="clean = the original protocol (split-at-lacuna segments, no real "
                         "gap ever in context -- every prior benchmark number used this); "
                         "whole = the realistic task (full AS-EDITED document, real lacunae "
                         "intact elsewhere in view while filling in one blank).")
    ap.add_argument("--n", type=int, default=200, help="samples per length")
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--lengths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default=None,
                    help="contaminated_*.json from leak_scan.py — drop those segments")
    ap.add_argument("--force-unk-metadata", action="store_true",
                    help="score with region/century always UNK, regardless of each record's "
                         "own value -- use for the with-vs-without ablation's 'without' side, "
                         "and MANDATORY for any comparison against Ithaca (digit-3 test split, "
                         "matching Ithaca's own convention): Ithaca has no metadata-conditioning "
                         "capability, so a fair comparison can't give this model an input it "
                         "structurally can't have.")
    ap.add_argument("--verbose", action="store_true",
                    help="print each example's gold/pred/CER live as it's scored, not just "
                         "the per-length summary at the end -- useful on CPU where a single "
                         "length can take minutes.")
    a = ap.parse_args()
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.expandvars(a.ckpt), device)
    if device.type == "cuda":
        model.cfg.attn_impl = "sdpa"     # short contexts; dense is faster than flex compile
    model.eval()
    if a.mode == "whole":
        from iphi import load_whole_full as load_iphi_whole
        from papyri import load_whole_full as load_papyri_whole
        load_fn = load_iphi_whole if a.domain == "iphi" else load_papyri_whole
        recs = [r for r in load_fn(split=a.split, min_len=50) if len(r["chars"]) <= 4096]
    else:
        load_fn = load_iphi if a.domain == "iphi" else load_papyri
        recs = [r for r in load_fn(split=a.split, min_len=50) if len(r["chars"]) <= 1500]
    if a.exclude:
        raw_bad = json.loads(Path(os.path.expandvars(a.exclude)).read_text())["contaminated"]
        # ids compared as strings: papyri TM ids can be compound ("79442 79443") and
        # int() crashes on them; leak_scan stored them from the same source field
        bad_pairs = {(str(x[0]), int(x[1])) for x in raw_bad}
        bad_ids = {str(x[0]) for x in raw_bad}
        n0 = len(recs)
        if a.mode == "whole":
            # whole-document records are unsegmented (seg=0): drop the document if ANY
            # of its clean-split segments was flagged as pretraining-contaminated
            recs = [r for r in recs if str(r["phi_id"]) not in bad_ids]
        else:
            recs = [r for r in recs if (str(r["phi_id"]), int(r["seg"])) not in bad_pairs]
        print(f"excluded {n0 - len(recs)} pretraining-contaminated "
              f"{'documents' if a.mode == 'whole' else 'segments'} ({len(recs)} remain)")
    rng = np.random.default_rng(1234)
    rng.shuffle(recs)
    eval_fn = eval_span_whole if a.mode == "whole" else eval_span
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        r = eval_fn(model, recs, L, device, a.n, a.beam, force_unk_metadata=a.force_unk_metadata,
                   verbose=a.verbose)
        rows.append(r)
        print(f"L={r['L']:>2}  CER={r['CER']:.4f}  top1={r['top1']:.4f}  "
              f"top20={r['top20']:.4f}  (n={r['n']})", flush=True)
    avg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("CER", "top1", "top20")}
    res = dict(ckpt=a.ckpt, split=a.split, domain=a.domain, mode=a.mode, n_per_L=a.n, beam=a.beam,
               force_unk_metadata=a.force_unk_metadata, per_L=rows, avg=avg)
    print(f"domain={a.domain} mode={a.mode} force_unk_metadata={a.force_unk_metadata} "
          f"AVG(1-10): CER={avg['CER']:.4f} top1={avg['top1']:.4f} top20={avg['top20']:.4f}")
    if a.mode == "clean":
        print("ITHACA:    CER=0.2630 top1=0.6180 top20=0.7830")
    else:
        print("NOTE: 'whole' mode has no prior comparable number -- every earlier benchmark "
              "(including Ithaca's own) used clean, lacuna-free context. This is the first "
              "run of the realistic task.")
    if a.split == "test" and os.environ.get("INSC_TEST_DIGIT", "3") == "3" \
            and not a.force_unk_metadata:
        print("NOTE: digit-3 test split matches Ithaca's own convention, but this run did "
              "NOT force metadata to UNK -- not a fair Ithaca comparison; rerun with "
              "--force-unk-metadata for that.")
    if a.out:
        Path(os.path.expandvars(a.out)).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
