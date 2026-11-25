"""Same-harness Ithaca baseline: run DeepMind's released Ithaca model on EXACTLY the
(segment, gap) samples that insc_eval/restore.py evaluates, scored letters-only with
the same CER/top-1/top-20. Every protocol delta then cancels in the comparison.

Sampling mirrors restore.py verbatim: load_iphi(split, min_len=50), len<=1500,
exclusion list, rng(1234) shuffle; per length L an rng(L) draws gap starts with the
same skip rule. Ithaca input: the same +-384-letter window, rendered as lowercase
unaccented text with spaces from the boundary plane (word-final sigma restored), the
gap letters AND gap-internal spaces replaced by '?' (their protocol knows the physical
lacuna width), trimmed to Ithaca's 768-char model window centered on the gap.
Predictions are space-stripped and sigma-folded back to the 24-letter space.

Run inside .venv-ithaca (jax CPU):
  python insc_eval/ithaca_baseline.py --split val --n 200 \
      --exclude $INS_DATA/contaminated_val_fold0.json --out $INS_DATA/runs/ithaca_val_clean.json
"""
from __future__ import annotations

import argparse, functools, json, os, pickle, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from data.normalize import ALPHABET
from iphi import load as load_iphi

ALIST = list(ALPHABET)
CKPT_DEFAULT = os.path.expandvars("$INS_DATA/ithaca_checkpoint.pkl")
ITHACA_TEXT_LEN = 768
FOLD = {"ς": "σ", "ϲ": "σ"}   # prediction -> 24-letter space


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


def build_sample(r, L, s, ctx=768):
    """Mirror restore.py's window; render Ithaca text + gold letters."""
    chars = np.asarray(r["chars"], np.int64)
    bnd = np.asarray(r["boundary"], np.int64)
    lo = max(0, s - ctx // 2); hi = min(len(chars), s + L + ctx // 2)
    window = chars[lo:hi]; wb = bnd[lo:hi]
    g0, g1 = s - lo, s - lo + L                       # gap letter span in window
    gold = "".join(ALIST[c] for c in window[g0:g1])

    pieces = []          # (char, is_gap) — flat text stream
    for i, (c, b) in enumerate(zip(window, wb)):
        in_gap = g0 <= i < g1
        ch = ALIST[c]
        if not in_gap and ch == "σ" and b >= 1:
            ch = "ς"                                   # word-final sigma for Ithaca
        pieces.append(("?" if in_gap else ch, in_gap))
        if b >= 1 and i < len(window) - 1:
            pieces.append(("?" if in_gap and i < g1 - 1 else " ", in_gap and i < g1 - 1))
    # trim to Ithaca's window, centered on the gap
    gidx = [k for k, (_, g) in enumerate(pieces) if g]
    lo_t = max(0, (gidx[0] + gidx[-1]) // 2 - (ITHACA_TEXT_LEN - 20) // 2)
    hi_t = min(len(pieces), lo_t + ITHACA_TEXT_LEN - 20)
    lo_t = max(0, min(lo_t, gidx[0]))                  # never cut the gap
    hi_t = max(hi_t, gidx[-1] + 1)
    text = "".join(ch for ch, _ in pieces[lo_t:hi_t]).strip()
    return text, gold


G = {}


def _init(ckpt_path):
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                       "intra_op_parallelism_threads=4")
    import jax
    from ithaca.eval import inference
    from ithaca.models.model import Model
    from ithaca.util.alphabet import GreekAlphabet
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    params = jax.device_put(checkpoint["params"])
    model = Model(**checkpoint["model_config"])
    forward = functools.partial(model.apply, params)
    alphabet = GreekAlphabet()
    alphabet.idx2word = checkpoint["alphabet"]["idx2word"]
    alphabet.word2idx = checkpoint["alphabet"]["word2idx"]
    G.update(inference=inference, forward=forward, params=params, alphabet=alphabet,
             cfg=checkpoint["model_config"])


def _restore_one(args):
    """One sample -> up to 20 letters-only gap hypotheses (best first)."""
    text, gold = args
    inference = G["inference"]
    try:
        # core of inference.restore() minus the saliency pass
        import jax
        import ithaca.util.eval as eval_util
        t, _, text_padded, _, _, _, _, restore_mask_idx = inference._prepare_text(
            text, G["alphabet"])
        beam = eval_util.beam_search_batch_2d(
            G["forward"], G["alphabet"], text_padded, restore_mask_idx,
            beam_width=inference.RESTORATION_BEAM_WIDTH,
            temperature=inference.RESTORATION_TEMPERATURE,
            rng=jax.random.PRNGKey(inference.SEED))
        idx = [i - 1 for i in restore_mask_idx]
        hyps = []
        for be in beam:
            full = be.text_pred[1:]
            pred = "".join(full[i] for i in idx if i < len(full))
            pred = "".join(FOLD.get(c, c) for c in pred if c not in " -?")
            hyps.append(pred)
        return gold, hyps
    except Exception as e:
        return gold, ["<ERR:%s>" % str(e)[:60]]


def _restore_one_strict(args):
    """Strict mode: keep spaces in the prediction; score in the shared char space."""
    text, gold = args
    g, hyps = _restore_one((text, gold))
    if hyps and hyps[0].startswith("<ERR"):
        return g, hyps
    return g, hyps


def run_strict(a):
    """Consume a frozen samples file; gap = '?'*L at [start, start+L); spaces count."""
    STRICT_FOLD = {"ς": "σ", "ϲ": "σ", "ϙ": "κ", "ϛ": "σ"}

    def canon(s):
        return "".join(STRICT_FOLD.get(c, c) for c in s)

    samples = json.loads(Path(os.path.expandvars(a.samples)).read_text())
    want = {int(x) for x in a.lengths.split(",")}
    samples = [s for s in samples if s["L"] in want]
    if a.shard:
        i, k = (int(x) for x in a.shard.split(","))
        samples = samples[i::k]
    tasks = [(s["text"][:s["start"]] + "?" * s["L"] + s["text"][s["start"] + s["L"]:],
              canon(s["gold"])) for s in samples]
    meta = [s["L"] for s in samples]
    print(f"{len(tasks)} strict tasks (lengths {sorted(want)})", flush=True)

    _init(a.ckpt)
    import time, jax
    import ithaca.util.eval as eval_util
    from ithaca.eval import inference
    results = []
    scores_all = []          # per-sample beam log-probs (same order as hyps)
    t0 = time.time()
    for i, (text, gold) in enumerate(tasks):
        try:
            _, _, tp, _, _, _, _, rmi = inference._prepare_text(text, G["alphabet"])
            beam = eval_util.beam_search_batch_2d(
                G["forward"], G["alphabet"], tp, rmi,
                beam_width=inference.RESTORATION_BEAM_WIDTH,
                temperature=inference.RESTORATION_TEMPERATURE,
                rng=jax.random.PRNGKey(inference.SEED))
            idx = [k - 1 for k in rmi]
            hyps = []
            for be in beam:
                full = be.text_pred[1:]
                raw = "".join(full[k] for k in idx if k < len(full))
                hyps.append(canon(raw))          # spaces KEPT — strict space
            results.append((gold, hyps))
            scores_all.append([round(float(be.pred_logprob), 5) for be in beam])
        except Exception as e:
            results.append((gold, ["<ERR:%s>" % str(e)[:60]]))
            scores_all.append([])
        if (i + 1) % 50 == 0:
            r = (time.time() - t0) / (i + 1)
            print(f"  {i+1}/{len(tasks)} ({r:.1f}s/sample)", flush=True)

    rows = {}
    n_err = 0
    for L, (gold, hyps) in zip(meta, results):
        r = rows.setdefault(L, dict(L=L, n=0, cers=[], t1=0, t20=0))
        if hyps and hyps[0].startswith("<ERR"):
            n_err += 1; continue
        pred = hyps[0] if hyps else ""
        r["cers"].append(levenshtein(pred, gold) / max(len(gold), 1))
        r["t1"] += int(pred == gold)
        r["t20"] += int(any(h == gold for h in hyps))
        r["n"] += 1
    out_rows = []
    for L in sorted(rows):
        r = rows[L]
        row = dict(L=L, n=r["n"], CER=round(float(np.mean(r["cers"])), 4),
                   top1=round(r["t1"] / max(r["n"], 1), 4),
                   top20=round(r["t20"] / max(r["n"], 1), 4))
        out_rows.append(row)
        print(f"L={L:>2}  CER={row['CER']:.4f}  top1={row['top1']:.4f}  "
              f"top20={row['top20']:.4f}  (n={row['n']})", flush=True)
    if a.out:
        Path(os.path.expandvars(a.out)).write_text(json.dumps(dict(
            model="ithaca_v1_release", protocol="strict", per_L=out_rows,
            errors=n_err,
            samples=[dict(phi_id=s["phi_id"], L=s["L"], start=s["start"], gold=g,
                          preds=h, scores=sc)
                     for s, (g, h), sc in zip(samples, results, scores_all)]), indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lengths", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--samples", default=None,
                    help="strict mode: frozen samples file from restore_strict.py")
    ap.add_argument("--shard", default=None, help="i,k -> process samples[i::k]")
    a = ap.parse_args()
    if a.samples:
        return run_strict(a)

    recs = [r for r in load_iphi(split=a.split, min_len=50) if len(r["chars"]) <= 1500]
    if a.exclude:
        bad = {tuple(x) for x in
               json.loads(Path(os.path.expandvars(a.exclude)).read_text())["contaminated"]}
        n0 = len(recs)
        recs = [r for r in recs if (int(r["phi_id"]), int(r["seg"])) not in bad]
        print(f"excluded {n0 - len(recs)} pretraining-contaminated segments "
              f"({len(recs)} remain)", flush=True)
    rng = np.random.default_rng(1234)
    rng.shuffle(recs)

    # mirror restore.py: same per-L RNG stream, same skip rule, first n usable records
    tasks, meta = [], []
    for L in [int(x) for x in a.lengths.split(",")]:
        lrng = np.random.default_rng(0 + L)
        tot = 0
        for r in recs:
            chars = r["chars"]
            if len(chars) <= L + 8:
                continue
            s = int(lrng.integers(4, len(chars) - L - 4))
            tasks.append(build_sample(r, L, s))
            meta.append(L)
            tot += 1
            if tot >= a.n:
                break
    print(f"{len(tasks)} restoration tasks; running Ithaca beam-20 "
          f"({a.workers} workers)", flush=True)

    if a.workers <= 1:
        # single-process path (GPU): no fork, incremental progress
        import time
        _init(a.ckpt)
        results = []
        t0 = time.time()
        for i, t in enumerate(tasks):
            results.append(_restore_one(t))
            if (i + 1) % 50 == 0:
                r = (time.time() - t0) / (i + 1)
                print(f"  {i+1}/{len(tasks)} ({r:.1f}s/sample, "
                      f"ETA {(len(tasks)-i-1)*r/60:.0f} min)", flush=True)
    else:
        with Pool(a.workers, initializer=_init, initargs=(a.ckpt,)) as pool:
            results = pool.map(_restore_one, tasks, chunksize=4)

    rows = {}
    n_err = 0
    for L, (gold, hyps) in zip(meta, results):
        r = rows.setdefault(L, dict(L=L, n=0, cers=[], t1=0, t20=0))
        if hyps and hyps[0].startswith("<ERR"):
            n_err += 1
            continue
        pred = hyps[0] if hyps else ""
        r["cers"].append(levenshtein(pred, gold) / max(len(gold), 1))
        r["t1"] += int(pred == gold)
        r["t20"] += int(any(h == gold for h in hyps))
        r["n"] += 1
    out_rows = []
    for L in sorted(rows):
        r = rows[L]
        row = dict(L=L, n=r["n"], CER=round(float(np.mean(r["cers"])), 4),
                   top1=round(r["t1"] / max(r["n"], 1), 4),
                   top20=round(r["t20"] / max(r["n"], 1), 4))
        out_rows.append(row)
        print(f"L={L:>2}  CER={row['CER']:.4f}  top1={row['top1']:.4f}  "
              f"top20={row['top20']:.4f}  (n={row['n']})", flush=True)
    avg = {k: round(float(np.mean([r[k] for r in out_rows])), 4)
           for k in ("CER", "top1", "top20")}
    print(f"AVG: CER={avg['CER']:.4f} top1={avg['top1']:.4f} top20={avg['top20']:.4f}"
          f"  (errors: {n_err})")
    if a.out:
        Path(os.path.expandvars(a.out)).write_text(json.dumps(dict(
            model="ithaca_v1_release", split=a.split, n_per_L=a.n,
            per_L=out_rows, avg=avg, errors=n_err), indent=1))


if __name__ == "__main__":
    main()
