"""STRICT Ithaca-protocol restoration eval — fully symmetric between models.

Protocol (no asymmetries):
  - unit: Ithaca's character space — SPACES COUNT. A gap of length L covers L
    consecutive characters (letters and/or spaces) of the inscription's ithaca_text.
  - context: the WHOLE inscription as edited, other lacunae ('-' runs) included as
    unknowns of stated width. length 50..767 (Ithaca's model window).
  - our model must PREDICT word division inside the gap (boundary head); division
    errors count in CER and break exact-match. Sigma variants folded (final sigma is
    deterministic given division).
  - CER = Levenshtein over the gap's char string (spaces incl.) / L; top-1/top-20
    exact over the same string. n per length, lengths 1..10, decontaminated test.

Modes:
  --make-samples: generate the frozen sample file both models consume.
      python insc_eval/restore_strict.py --make-samples --split test --n 300 \
        --exclude $INS_DATA/contaminated_test_fold0.json --samples $INS_DATA/strict_test_samples.json
  --ckpt: evaluate OUR model on a samples file.
      python insc_eval/restore_strict.py --ckpt <best.pt> --samples ... --out ...
(The Ithaca side consumes the same samples file via ithaca_baseline.py --samples.)
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from data.normalize import ALPHABET

ALIST = list(ALPHABET)
A_IDX = {c: i for i, c in enumerate(ALIST)}
RAW_JSONL = os.path.expandvars("$INS_DATA/raw/iphi.jsonl")
MASK, UNK_BND, UNK_DIA, UNK_PUNCT = 24, 3, 48, 6
# metadata-conditioned checkpoints (finetune_whole.py) read region/century inputs; this
# harness always feeds UNK -- Ithaca predicts date/region rather than consuming them, so
# giving ours the true values would break the symmetry of the comparison.
UNK_REGION, UNK_CENTURY = 14, 15
FOLD = {"ς": "σ", "ϲ": "σ", "ϙ": "κ", "ϛ": "σ"}


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


def canon(s):
    return "".join(FOLD.get(c, c) for c in s)


def load_inscriptions(split, exclude=None):
    """Whole inscriptions: ithaca_text (lowercase accentless, spaces, '-' damage)."""
    bad_phi = set()
    if exclude:
        j = json.loads(Path(os.path.expandvars(exclude)).read_text())
        bad_phi = {int(x[0]) for x in j["contaminated"]
                   if str(x[0]) != "POSITIVE_CONTROL"}   # inscription-level exclusion
    out = []
    with open(RAW_JSONL) as f:
        for line in f:
            r = json.loads(line)
            pid = int(r["PHI_ID"])
            test_d = os.environ.get("INSC_TEST_DIGIT", "3")
            val_d = os.environ.get("INSC_VAL_DIGIT", "4")
            sp = {test_d: "test", val_d: "val"}.get(str(pid)[-1], "train")
            if sp != split or pid in bad_phi:
                continue
            t = (r.get("ithaca_text") or "").strip().lower()
            t = " ".join(t.split())
            t = canon(t)
            if 50 <= len(t) <= 767:
                out.append(dict(phi_id=pid, text=t))
    return out


def make_samples(split, n, lengths, exclude, out_path):
    recs = load_inscriptions(split, exclude)
    rng = np.random.default_rng(1234)
    rng.shuffle(recs)
    print(f"{len(recs)} whole inscriptions in {split} (decontaminated, 50-767 chars)")
    samples = []
    for L in lengths:
        lrng = np.random.default_rng(0 + L)
        tot = 0
        for r in recs:
            t = r["text"]
            if len(t) <= L + 8:
                continue
            s = int(lrng.integers(4, len(t) - L - 4))
            gap = t[s:s + L]
            # gap must be attested text: no '-' damage, no double space, not
            # space-adjacent-to-space at the edges (unscorable), >=1 letter
            if ("-" in gap or "  " in gap or gap.strip(" ") == "" or
                    (s > 0 and t[s - 1] == " " and gap[0] == " ") or
                    (t[s + L:s + L + 1] == " " and gap[-1] == " ")):
                continue
            samples.append(dict(phi_id=r["phi_id"], L=L, start=s, gold=gap, text=t))
            tot += 1
            if tot >= n:
                break
        print(f"L={L}: {tot} samples")
    Path(os.path.expandvars(out_path)).write_text(json.dumps(samples))
    print(f"wrote {len(samples)} samples -> {out_path}")


# ---------------- our model: joint boundary+letter beam in their char space -------

def text_to_planes(t):
    """ithaca_text -> letter ids, boundary flags (1 = space after), letter->char pos."""
    ids, bnd, pos = [], [], []
    for i, ch in enumerate(t):
        if ch == " ":
            if bnd:
                bnd[-1] = 1
        elif ch == "-":
            ids.append(MASK); bnd.append(UNK_BND); pos.append(i)
        elif ch in A_IDX:
            ids.append(A_IDX[ch]); bnd.append(0); pos.append(i)
        # any other char: skip (ithaca_text is clean)
    return np.array(ids, np.int64), np.array(bnd, np.int64), pos


def space_patterns(L, prev_is_letter, next_exists, max_spaces=3):
    """All ways to place spaces among the L gap slots (internal Greek: no '  ')."""
    from itertools import combinations
    slots = list(range(L))
    pats = []
    for k in range(0, min(max_spaces, L) + 1):
        for c in combinations(slots, k):
            if any(b - a == 1 for a, b in zip(c, c[1:])):
                continue                      # no double space
            if L - k == 0:
                continue                      # at least one letter
            pats.append(tuple(c))
    return pats


def eval_ours(ckpt, samples_path, out_path, beam_width=20, ctx=768, pat_keep=4,
              lengths_filter=None, shard=None):
    import torch
    from eval.intrinsic import load_model
    sys.path.insert(2, str(Path(__file__).resolve().parent))
    from restore import beam_restore, _char_logp   # reuse the letter beam
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(os.path.expandvars(ckpt), device)
    model.cfg.attn_impl = "sdpa"
    model.eval()
    samples = json.loads(Path(os.path.expandvars(samples_path)).read_text())
    if lengths_filter:
        samples = [s for s in samples if s["L"] in lengths_filter]
    if shard:
        i, k = (int(x) for x in shard.split(","))
        samples = samples[i::k]
    rows = {}
    per_sample = []          # per-sample top-20 + scores (error analysis, calibration)
    import time
    t0 = time.time()
    @torch.no_grad()
    def fill_forward(ids_list, bnd_row, free_idx):
        """Batched forward on FILLED candidates; returns log P(word-end) per free site."""
        B = len(ids_list); T = len(ids_list[0])
        batch = dict(
            input_ids=torch.tensor(np.stack(ids_list), dtype=torch.long, device=device),
            boundary=torch.tensor(bnd_row, dtype=torch.long, device=device)[None].expand(B, T).contiguous(),
            dia=torch.full((B, T), UNK_DIA, dtype=torch.long, device=device),
            punct=torch.full((B, T), UNK_PUNCT, dtype=torch.long, device=device),
            region=torch.full((B, T), UNK_REGION, dtype=torch.long, device=device),
            century=torch.full((B, T), UNK_CENTURY, dtype=torch.long, device=device),
            seg_id=torch.ones(B, T, dtype=torch.long, device=device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ob = model(batch)["boundary"]
        pb = torch.softmax(ob.float(), -1)
        p_end = (pb[:, :, 1:3].sum(-1)).clamp(1e-4, 1 - 1e-4)   # word-end or sent-end
        return p_end[:, free_idx].log().cpu().numpy(), (1 - p_end)[:, free_idx].log().cpu().numpy()

    A_STR = "".join(ALIST)
    for si, smp in enumerate(samples):
        t, s, L, gold = smp["text"], smp["start"], smp["L"], smp["gold"]
        pre, post = t[:s], t[s + L:]
        lead_ok = not pre.endswith(" ") and len(pre) > 0
        trail_ok = not post.startswith(" ") and len(post) > 0
        hyps = {}
        for k in range(max(1, L - 3), L + 1):
            n_sp = L - k
            # sites: [leading] + after each of the k letters (last = trailing)
            n_sites = k + 1
            if n_sp > (1 if lead_ok else 0) + (k - 1) + (1 if trail_ok else 0):
                continue
            full = pre + "\u03b1" * k + post          # k placeholder letters, no spaces
            ids, bnd, pos = text_to_planes(full)
            npre = sum(1 for q in pos if q < s)
            gidx = list(range(npre, npre + k))
            ids2 = ids.copy(); ids2[gidx] = MASK
            bnd2 = np.where(bnd == UNK_BND, UNK_BND, np.minimum(bnd, 2))
            bnd2[gidx] = UNK_BND                       # free division inside the gap
            if lead_ok and npre > 0:
                bnd2[npre - 1] = UNK_BND               # leading-space site is free too
            # context parity holds by construction: samples are whole inscriptions
            # capped at 767 THEIR-chars (make_samples), i.e. both models always see
            # the complete text — this letter-window is never the binding constraint
            lo = max(0, gidx[0] - ctx // 2); hi = min(len(ids2), gidx[-1] + ctx // 2)
            window = ids2[lo:hi].copy()
            gap_rel = [g - lo for g in gidx]
            bw = bnd2[lo:hi].copy()
            with torch.no_grad():
                cand = beam_restore(model, window, gap_rel, bw, device, beam_width)
            if not cand:
                continue
            cand = cand[:8]                            # fill-scored candidates per k
            filled = []
            for txt24, _ in cand:
                w2 = window.copy()
                for j, chs in zip(gap_rel, txt24):
                    w2[j] = A_STR.index(chs)
                filled.append(w2)
            # free sites (window-relative letter indices): pre letter + gap letters
            site_idx = ([gap_rel[0] - 1] if lead_ok and gap_rel[0] > 0 else []) + gap_rel
            lp_end, lp_no = fill_forward(filled, bw, site_idx)
            has_lead = bool(site_idx and lead_ok and gap_rel[0] > 0 and
                            site_idx[0] == gap_rel[0] - 1)
            for ci, (txt24, bscore) in enumerate(cand):
                pe, pn = lp_end[ci], lp_no[ci]
                # valid site positions in the site list
                valid = []
                for vi in range(len(site_idx)):
                    if vi == 0 and has_lead:
                        valid.append(vi)               # leading space
                    elif vi == len(site_idx) - 1:
                        if trail_ok:
                            valid.append(vi)           # trailing space
                    elif not (vi == 0 and has_lead):
                        valid.append(vi)               # space between letters
                order = sorted(valid, key=lambda v: -pe[v])
                chosen = set(order[:n_sp]) if n_sp else set()
                if len(chosen) < n_sp:
                    continue
                arr_lp = sum(pe[v] for v in chosen) + \
                         sum(pn[v] for v in valid if v not in chosen)
                total = bscore + arr_lp
                norm = total / (k + len(valid))
                # render into the shared char space
                out_str = ""
                li = 0
                for vi in range(len(site_idx)):
                    if vi == 0 and has_lead:
                        if vi in chosen:
                            out_str += " "
                        continue
                    out_str += txt24[li]; li += 1
                    if vi in chosen:
                        out_str += " "
                if len(out_str) != L or li != len(txt24):
                    continue                            # malformed render: reject
                if out_str not in hyps or hyps[out_str] < norm:
                    hyps[out_str] = norm
        ranked = sorted(hyps.items(), key=lambda kv: -kv[1])[:beam_width]
        preds = [k for k, _ in ranked]
        per_sample.append(dict(phi_id=smp["phi_id"], L=L, start=s, gold=canon(gold),
                               preds=preds, scores=[round(float(v), 5) for _, v in ranked]))
        gold_c = canon(gold)
        r = rows.setdefault(L, dict(L=L, n=0, cers=[], t1=0, t20=0, cers_let=[]))
        pred = preds[0] if preds else ""
        r["cers"].append(levenshtein(pred, gold_c) / max(len(gold_c), 1))
        # letters-only CER: same prediction, spaces stripped from both sides. The char CER
        # above is the Ithaca-comparable number, but ~15% of every gap is word breaks,
        # which are far easier to predict than letters -- so char CER is NOT comparable to
        # the whole-document eval (where all L positions are letters). This column is.
        gl = gold_c.replace(" ", "")
        r["cers_let"].append(levenshtein(pred.replace(" ", ""), gl) / max(len(gl), 1))
        r["t1"] += int(pred == gold_c)
        r["t20"] += int(any(p == gold_c for p in preds))
        r["n"] += 1
        if (si + 1) % 50 == 0:
            rate = (time.time() - t0) / (si + 1)
            print(f"  {si+1}/{len(samples)} ({rate:.1f}s/sample)", flush=True)
            _write_report(rows, ckpt, out_path, n_done=si + 1, n_total=len(samples),
                          per_sample=per_sample)
    _write_report(rows, ckpt, out_path, n_done=len(samples), n_total=len(samples), final=True,
                  per_sample=per_sample)


def _write_report(rows, ckpt, out_path, n_done, n_total, final=False, per_sample=None):
    """Write the current per-L / avg report -- called periodically during the sample
    loop (not just at the very end) so a time-limit kill never loses everything, same
    fix as fold_crux.py's incremental checkpointing."""
    out_rows = []
    for L in sorted(rows):
        r = rows[L]
        row = dict(L=L, n=r["n"], CER=round(float(np.mean(r["cers"])), 4),
                   CER_letters=round(float(np.mean(r["cers_let"])), 4)
                   if r.get("cers_let") else None,
                   top1=round(r["t1"] / max(r["n"], 1), 4),
                   top20=round(r["t20"] / max(r["n"], 1), 4))
        out_rows.append(row)
        if final:
            print(f"L={L:>2}  CER={row['CER']:.4f}  CERlet={row['CER_letters']:.4f}  "
                  f"top1={row['top1']:.4f}  top20={row['top20']:.4f}  (n={row['n']})",
                  flush=True)
    avg = {k: round(float(np.mean([r[k] for r in out_rows if r[k] is not None])), 4)
           for k in ("CER", "CER_letters", "top1", "top20")}
    tag = "STRICT AVG" if final else f"[checkpoint {n_done}/{n_total}]"
    print(f"{tag}: CER={avg['CER']:.4f} CERlet={avg['CER_letters']:.4f} "
          f"top1={avg['top1']:.4f} top20={avg['top20']:.4f}", flush=True)
    Path(os.path.expandvars(out_path)).write_text(json.dumps(dict(
        ckpt=ckpt, protocol="strict", per_L=out_rows, avg=avg,
        n_done=n_done, n_total=n_total, complete=final,
        **({"samples": per_sample} if per_sample is not None else {})), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-samples", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lengths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--shard", default=None, help="i,k -> process samples[i::k]")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    if a.make_samples:
        make_samples(a.split, a.n, lengths, a.exclude, a.samples)
    elif a.ckpt:
        eval_ours(a.ckpt, a.samples, a.out, beam_width=a.beam,
                  lengths_filter=set(lengths), shard=a.shard)
    else:
        raise SystemExit("need --make-samples or --ckpt")


if __name__ == "__main__":
    main()
