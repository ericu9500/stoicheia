"""Inference + benchmark glue for a trained MeterModel.

  # Norma benchmark, both tasks, scored in-process (--norma-source hf|git, default hf):
  python -m meter.predict --model $STOICHEIA_DATA/runs/meter_joint/best.pt --norma \
      [--pred-out data/norma_preds.jsonl] [--norma-source hf]
  # work-split scanner dev/test:
  python -m meter.predict --model ... --scan-split
  # production:
  python -m meter.predict --model ... --macronize in.txt out.txt
  python -m meter.predict --model ... --scan in.txt out.txt

--norma scores both tasks directly via meter.norma_score (acc, balanced acc, per-class
F1 for macronize; acc, balanced acc, boundary-F1, weight acc for syllabify), split into
dev/test, and additionally dumps macron predictions/probabilities in the old
predictions-jsonl format for anyone still using the legacy ensemble/scorer scripts.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meter.backbone import load_backbone
from meter.dataset import batch_rows, encode_plain, load_records, pack_records
from meter.marks import (ambiguous_mask, bracketize, enforce_circumflex_heavy,
                         insert_marks, merge_vowelless_syllables,
                         parse_macron_line, parse_scan_line)
from meter.model import MeterConfig, MeterModel
from meter.norma_data import add_norma_source_arg
from meter.norma_score import scan_metrics
from meter.train import SCAN_DEV_WORKS, SCAN_TEST_WORKS


def load_model(path, device, attn="sdpa"):
    sd = torch.load(path, map_location="cpu")
    encoder, _ = load_backbone(os.path.expandvars(sd["cfg"]["ckpt"]), device, attn)
    mcfg = MeterConfig(**sd["mcfg"])
    model = MeterModel(encoder, mcfg).to(device)
    # the released Hub checkpoints omit the class-weight buffers (constants fixed by mcfg): tolerate that only
    res = model.load_state_dict(sd["model"], strict=False)
    assert set(res.missing_keys) <= {"mac_w", "scan_w"} and not res.unexpected_keys, (res.missing_keys, res.unexpected_keys)
    model.eval()
    return model, sd


@torch.no_grad()
def predict_records(model, records, T, micro, device):
    """-> per record (mac_argmax, scan_argmax, mac_P(long), scan_probs) with None
    entries preserved. mac_P(long) is P(class 0) per letter; scan_probs is (n,4).
    scan_argmax has two deterministic corrections applied, in order: (1)
    merge_vowelless_syllables -- a predicted syllable span with no vowel gets
    folded into the preceding one, keeping its own (usually already-correct)
    weight; (2) enforce_circumflex_heavy -- a circumflexed syllable is always
    heavy. Every consumer (scoring, --scan production output, Viterbi) gets both
    fixed rules for free."""
    live = [i for i, r in enumerate(records) if r is not None]
    rows, skipped = pack_records(records, T, live)
    if skipped:
        print(f"  WARNING: {skipped} records longer than T={T} skipped", file=sys.stderr)
    out = [None] * len(records)
    for i in range(0, len(rows), micro):
        chunk = rows[i:i + micro]
        batch = batch_rows(chunk, records, T, device=device, with_slots=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model({k: v for k, v in batch.items() if k != "slots"})
        mac_p = torch.softmax(o["mac"].float(), -1).cpu().numpy()
        scan_p = torch.softmax(o["scan"].float(), -1).cpu().numpy()
        pm = mac_p.argmax(-1)
        ps = scan_p.argmax(-1)
        for b, slots in enumerate(batch["slots"]):
            for ri, c in slots:
                n = len(records[ri])
                scan_argmax = merge_vowelless_syllables(records[ri].chars, ps[b, c:c + n])
                scan_argmax = enforce_circumflex_heavy(records[ri].dia, scan_argmax)
                out[ri] = (pm[b, c:c + n], scan_argmax,
                           mac_p[b, c:c + n, 0], scan_p[b, c:c + n])
    return out


def run_norma(model, sd, device, micro, pred_out, norma_source="hf"):
    from meter.norma_data import load_norma
    from meter.norma_score import MAC_LONG, mac_metrics
    T = sd["T"]
    norma = load_norma(norma_source)

    # -------- macronize: scored directly (in-process; no external scorer needed)
    probs_rows = []
    with open(pred_out, "w", encoding="utf-8") as f:
        for split_name, rows in [("dev", norma["dev"]), ("test", norma["test"])]:
            mac_rows = [d for d in rows if d["task"] == "macronize"]
            items = [(d["source"], *parse_macron_line(d["text"])) for d in mac_rows]
            recs = [encode_plain(p) for _, p, _ in items]
            preds = predict_records(model, recs, T, micro, device)
            pairs, by_source = [], defaultdict(list)
            for (srcname, plain, gold), pr in zip(items, preds):
                g = sorted(gold.items())
                pred_labels = [(k, int(pr[0][k]) if pr is not None else MAC_LONG) for k, _ in g]
                f.write(json.dumps(dict(
                    split=split_name, source=srcname,
                    gold=[[k, "", v] for k, v in g],
                    pred=[[k, "", pv] for k, pv in pred_labels]), ensure_ascii=False) + "\n")
                if pr is not None:
                    probs_rows.append(dict(split=split_name, source=srcname, seq=[
                        [v, float(pr[2][k])] for k, v in g]))
                for (k, gv), (_, pv) in zip(g, pred_labels):
                    pairs.append((gv, pv))
                    by_source[srcname].append((gv, pv))
            overall = mac_metrics(pairs)
            per_source = {s: mac_metrics(ps) for s, ps in by_source.items()}
            bals = [m["bal_acc"] for m in per_source.values() if m and m["bal_acc"] is not None]
            macro = round(sum(bals) / len(bals), 4) if bals else None
            print(f"macron {split_name}: {json.dumps(dict(**overall, macro_bal_acc=macro))}"
                  if overall else f"macron {split_name}: (empty)")
    probs_out = str(pred_out).replace(".jsonl", "") + "_probs.json"
    json.dump(probs_rows, open(probs_out, "w"))
    print(f"macron P(long) dump -> {probs_out}")
    print(f"macron preds -> {pred_out}")

    # -------- syllabify: test only (Norma has no syllabify dev rows, on hf or git)
    syl_items, syl_recs = [], []
    for d in norma["test"]:
        if d["task"] != "syllabify":
            continue
        parsed = parse_scan_line(d["text"])
        if parsed is None:
            continue
        plain, gold = parsed
        rec = encode_plain(plain)
        if rec is None:
            continue
        syl_items.append((d["source"], gold))
        syl_recs.append(rec)
    preds = predict_records(model, syl_recs, T, micro, device)
    print("\n=== norma syllabify (test) ===")
    pairs, by_src = [], defaultdict(list)
    for (srcname, gold), pr in zip(syl_items, preds):
        if pr is None:
            continue
        golds = np.zeros(len(pr[1]), dtype=np.int64)
        for k, g in gold.items():
            golds[k] = g
        rec_pairs = list(zip(golds.tolist(), pr[1].tolist()))
        pairs += rec_pairs
        by_src[srcname] += rec_pairs
    m = scan_metrics(pairs)
    bals = [b["bal_acc"] for s in sorted(by_src) if (b := scan_metrics(by_src[s]))]
    macro = round(sum(bals) / len(bals), 4) if bals else None
    print(f"  {json.dumps(dict(**m, macro_bal_acc=macro))}")


def run_scan_split(model, sd, device, micro):
    T = sd["T"]
    enc_dir = Path(os.path.expandvars(sd["cfg"]["encoded"]))
    recs, works = load_records(enc_dir / "scan_corpus.npz")
    for split, wanted in (("dev", SCAN_DEV_WORKS), ("test", SCAN_TEST_WORKS)):
        sel = [(r, w) for r, w in zip(recs, works) if w in wanted]
        preds = predict_records(model, [r for r, _ in sel], T, micro, device)
        pairs, by_work = [], defaultdict(list)
        for (r, w), pr in zip(sel, preds):
            if pr is None:
                continue
            p = list(zip(r.y_scan.tolist(), pr[1].tolist()))
            p = [(g, q) for g, q in p if g != -100]
            pairs += p
            by_work[w] += p
        print(f"\n=== scan {split} (whole verses, by work) ===")
        print(f"  all: {json.dumps(scan_metrics(pairs))}")
        for w in sorted(by_work):
            print(f"  {w}: {json.dumps(scan_metrics(by_work[w]))}")


def run_viterbi(model, sd, src, device, micro, theta, norma_source="hf"):
    """Meter-constrained decoding: exact-line + char accuracy, raw argmax vs
    gated Viterbi (meter forced from the corpus label when a grammar exists,
    else auto-detected; Norma syllabify is always auto)."""
    import numpy as np
    from meter.viterbi import METER_MAP, gated_auto, gated_decode
    T = sd["T"]

    def decode_set(name, items):
        """items: (group, meter_name|None, gold {ord: lab}, record)"""
        recs = [r for _, _, _, r in items]
        preds = predict_records(model, recs, T, micro, device)
        agg = defaultdict(lambda: np.zeros(6, np.int64))
        # [lines, exact_raw, exact_vit, char_ok_raw, char_ok_vit, chars] per group
        applied = 0
        for (group, mname, gold, rec), pr in zip(items, preds):
            if pr is None:
                continue
            n = len(pr[3])
            golds = np.zeros(n, np.int64)
            for k, g in gold.items():
                golds[k] = g
            logp = np.log(np.clip(pr[3], 1e-9, 1.0))
            raw = pr[1]   # pre-computed argmax, with enforce_circumflex_heavy already applied
            grammar = METER_MAP.get(mname or "")
            if grammar:
                vit, ok = gated_decode(logp.tolist(), grammar, theta)
            else:
                _, vit, ok = gated_auto(logp.tolist(), theta)
            applied += ok
            vit = np.asarray(vit)
            for g in (group, "ALL"):
                a = agg[g]
                a[0] += 1
                a[1] += int((raw == golds).all())
                a[2] += int((vit == golds).all())
                a[3] += int((raw == golds).sum())
                a[4] += int((vit == golds).sum())
                a[5] += n
        print(f"\n=== viterbi {name} (theta={theta}, applied {applied}/"
              f"{agg['ALL'][0]}) ===")
        for g in sorted(agg, key=lambda x: (x != "ALL", x)):
            a = agg[g]
            print(f"  {g}: lines={a[0]} exact raw={a[1]/a[0]:.3f} "
                  f"vit={a[2]/a[0]:.3f} | char raw={a[3]/a[5]:.4f} vit={a[4]/a[5]:.4f}")

    # ---- work-split dev/test from the scanner corpus (grouped by grammar)
    from meter.marks import parse_scan_line
    for split, works in (("scan-dev", SCAN_DEV_WORKS), ("scan-test", SCAN_TEST_WORKS)):
        items = []
        for line in open(src / "data/scanner/corpus_v3.tsv", encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or parts[0] not in works:
                continue
            parsed = parse_scan_line(parts[2])
            if parsed is None:
                continue
            rec = encode_plain(parsed[0])
            if rec is None:
                continue
            grammar = METER_MAP.get(parts[1], f"auto({parts[1] or 'lyric'})")
            items.append((grammar, parts[1], parsed[1], rec))
        decode_set(split, items)

    # ---- Norma syllabify (auto meter; test only -- Norma has no syllabify dev rows)
    from meter.norma_data import load_norma
    items = []
    for d in load_norma(norma_source)["test"]:
        if d["task"] != "syllabify":
            continue
        parsed = parse_scan_line(d["text"])
        if parsed is None:
            continue
        rec = encode_plain(parsed[0])
        if rec is None:
            continue
        items.append((d["source"], None, parsed[1], rec))
    decode_set("norma-syllabify", items)


def run_file(model, sd, device, micro, mode, infile, outfile):
    T = sd["T"]
    raw = [l.rstrip("\n") for l in open(infile, encoding="utf-8")]
    plains = [parse_macron_line(l)[0] for l in raw]   # strips any existing marks
    recs = [encode_plain(p) if p.strip() else None for p in plains]
    preds = predict_records(model, recs, T, micro, device)
    with open(outfile, "w", encoding="utf-8") as f:
        for plain, rec, pr in zip(plains, recs, preds):
            if rec is None or pr is None:
                f.write(plain + "\n")
                continue
            if mode == "macronize":
                amb = ambiguous_mask(rec.chars, rec.boundary, rec.dia)
                labels = {int(i): int(pr[0][i]) for i in np.flatnonzero(amb)}
                f.write(insert_marks(plain, labels) + "\n")
            else:
                labels = {i: int(c) for i, c in enumerate(pr[1]) if c > 0}
                f.write(bracketize(plain, labels) + "\n")
    print(f"{mode}: {len(raw)} lines -> {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--norma", action="store_true")
    ap.add_argument("--pred-out", default=None)
    ap.add_argument("--scan-split", action="store_true")
    ap.add_argument("--viterbi", action="store_true")
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--macronize", nargs=2, metavar=("IN", "OUT"))
    ap.add_argument("--scan", nargs=2, metavar=("IN", "OUT"))
    add_norma_source_arg(ap)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sd = load_model(a.model, device, a.attn)
    src = Path(os.path.expandvars(os.environ.get("MACRONIZER_SRC",
                              "$MACRONIZER_SRC")))
    if a.norma:
        pred_out = a.pred_out or (Path(sd["cfg"]["out_dir"]) / "norma_pred.jsonl")
        run_norma(model, sd, device, a.micro, pred_out, a.norma_source)
    if a.scan_split:
        run_scan_split(model, sd, device, a.micro)
    if a.viterbi:
        run_viterbi(model, sd, src, device, a.micro, a.theta, a.norma_source)
    if a.macronize:
        run_file(model, sd, device, a.micro, "macronize", *a.macronize)
    if a.scan:
        run_file(model, sd, device, a.micro, "scan", *a.scan)


if __name__ == "__main__":
    main()
