"""Per-fold checkpoint selection on dev CER, using numbers already in the training log.

The v3 runs selected best.pt on bits_per_char, but bpc and dev CER decorrelate late in
training (r~0.82) -- bpc bottoms out and rises while CER keeps improving. So best.pt can
be ~0.02 dev CER worse than the run's end state. Both best.pt and last.pt are legitimate
checkpoints; choosing between them on DEV CER is standard model selection (dev is exactly
what selection is for) and costs no GPU, because every eval's dev CER is in the log.

  python3 analysis/pick_ckpt.py                # report
  python3 analysis/pick_ckpt.py --write        # write chosen_ckpt.txt into each run dir
"""
import glob, json, os, re, sys
I=os.path.expandvars("$STOICHEIA_DATA/insc_data/runs")
write="--write" in sys.argv
print(f"{'run':<28}{'best.pt cer':>12}{'last.pt cer':>12}  choice")
for d in sorted(glob.glob(f"{I}/whole_v*")):
    ev=os.path.join(d,"eval.jsonl")
    if not os.path.exists(ev): continue
    rows=[json.loads(l) for l in open(ev)]
    if not rows: continue
    # best.pt = argmin bpc (how these runs were selected); last.pt = final eval
    bb=min(rows,key=lambda r:r["bits_per_char"])
    last=rows[-1]
    pick="best.pt" if bb["dev_cer"]<=last["dev_cer"] else "last.pt"
    gain=abs(bb["dev_cer"]-last["dev_cer"])
    flag="" if pick=="best.pt" else f"   (+{gain:.4f} CER recovered)"
    print(f"{os.path.basename(d):<28}{bb['dev_cer']:>12.4f}{last['dev_cer']:>12.4f}  {pick}{flag}")
    if write:
        open(os.path.join(d,"chosen_ckpt.txt"),"w").write(pick+"\n")
