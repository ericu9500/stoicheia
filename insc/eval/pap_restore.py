"""PAPYRI (DDbDP) restoration eval — TM digit split (3=test/4=val); otherwise
identical to restore.py (inscriptions).

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
from papyri import load as load_iphi

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
def _char_logp(model, seqs, bnd_row, device):
    """seqs: list of same-length int arrays. Boundary known OUTSIDE the gap (stone
    preserves word dividers around a lacuna); dia/punct unknown. -> (B,T,24) log-probs."""
    B, T = len(seqs), len(seqs[0])
    ids = torch.tensor(np.stack(seqs), dtype=torch.long, device=device)
    bnd = torch.tensor(bnd_row, dtype=torch.long, device=device)[None].expand(B, T).contiguous()
    batch = dict(input_ids=ids, boundary=bnd,
                 dia=torch.full((B, T), UNK_DIA, dtype=torch.long, device=device),
                 punct=torch.full((B, T), UNK_PUNCT, dtype=torch.long, device=device),
                 seg_id=torch.ones(B, T, dtype=torch.long, device=device))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch)
    return torch.log_softmax(out["char"][:, :, :NLET].float(), -1)


@torch.no_grad()
def beam_restore(model, chars, gap, bnd_row, device, beam_width=20, expand=48):
    base = np.asarray(chars, dtype=np.int64)
    beam = [(base.copy(), tuple(gap), 0.0)]
    finished = {}
    while beam:
        logp = _char_logp(model, [h[0] for h in beam], bnd_row, device)
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


def eval_span(model, recs, L, device, n, beam_width=20, seed=0, ctx=768):
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
        cand = beam_restore(model, window, gap, bnd, device, beam_width)
        if not cand:
            continue
        pred = cand[0][0]
        cers.append(levenshtein(pred, gold) / max(len(gold), 1))
        t1 += int(pred == gold)
        t20 += int(any(c[0] == gold for c in cand))
        tot += 1
        if tot >= n:
            break
    return dict(L=L, n=tot, CER=round(float(np.mean(cers)), 4),
                top1=round(t1 / max(tot, 1), 4), top20=round(t20 / max(tot, 1), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=200, help="samples per length")
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--lengths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", default=None,
                    help="contaminated_*.json from leak_scan.py — drop those segments")
    a = ap.parse_args()
    import os
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.expandvars(a.ckpt), device)
    if device.type == "cuda":
        model.cfg.attn_impl = "sdpa"     # short contexts; dense is faster than flex compile
    model.eval()
    recs = [r for r in load_iphi(split=a.split, min_len=50) if len(r["chars"]) <= 1500]
    if a.exclude:
        bad = {(str(x[0]), int(x[1])) for x in
               json.loads(Path(os.path.expandvars(a.exclude)).read_text())["contaminated"]}
        n0 = len(recs)
        recs = [r for r in recs if (str(r["phi_id"]), int(r["seg"])) not in bad]
        print(f"excluded {n0 - len(recs)} pretraining-contaminated segments "
              f"({len(recs)} remain)")
    rng = np.random.default_rng(1234)
    rng.shuffle(recs)
    rows = []
    for L in [int(x) for x in a.lengths.split(",")]:
        r = eval_span(model, recs, L, device, a.n, a.beam)
        rows.append(r)
        print(f"L={r['L']:>2}  CER={r['CER']:.4f}  top1={r['top1']:.4f}  "
              f"top20={r['top20']:.4f}  (n={r['n']})", flush=True)
    avg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("CER", "top1", "top20")}
    res = dict(ckpt=a.ckpt, split=a.split, n_per_L=a.n, beam=a.beam, per_L=rows, avg=avg)
    print(f"AVG(1-10): CER={avg['CER']:.4f} top1={avg['top1']:.4f} top20={avg['top20']:.4f}")
    print("ITHACA:    CER=0.2630 top1=0.6180 top20=0.7830")
    if a.out:
        Path(os.path.expandvars(a.out)).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
