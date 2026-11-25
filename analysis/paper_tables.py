"""Regenerate every paper table from results already on disk. Reads only; safe to run
at any time and shows partial state (n=) while jobs are still finishing.

  python3 analysis/paper_tables.py           # all three experiments
  python3 analysis/paper_tables.py parsing   # documentary | parsing | meter
"""
from __future__ import annotations
import collections, glob, json, os, re, sys
import numpy as np

GCB=os.path.expandvars("$STOICHEIA_DATA")
INSC=os.path.expandvars("$STOICHEIA_DATA")
REPO=os.path.expandvars("$STOICHEIA_ROOT")

def _agg(v): return (np.mean(v), np.std(v), len(v)) if v else (float("nan"), 0.0, 0)

def parsing():
    print("\n"+"="*78); print("EXPERIMENT 2 -- PARSING (joint tagger+parser, OGA 5 folds x 2 seeds)"); print("="*78)
    g=collections.defaultdict(list)
    for d in sorted(glob.glob(f"{GCB}/parser_data/runs/joint_*_f*_s*")):
        p=os.path.join(d,"test_scores_greedy.json")
        if not os.path.exists(p): continue
        j=json.load(open(p))
        if j.get("las") is None: continue
        tag=re.sub(r"_f\d+_s\d+$","",os.path.basename(d))
        # joint_tune_* are single-fold LR-sweep diagnostics -- reported by lr_control(),
        # not systems for the main table (an n=1 row next to n=10 rows invites misreading)
        if tag.startswith("joint_tune_"): continue
        g[tag].append((j["las"], j["uas"], j["xpos_exact"], j["upos"]))
    if not g: print("  (no results yet)"); return
    print(f"{'system':<24}{'n':>3}  {'LAS':>15} {'UAS':>8} {'XPOS':>8} {'UPOS':>8}")
    for tag,v in sorted(g.items(), key=lambda kv:-np.mean([x[0] for x in kv[1]])):
        m,s,n=_agg([x[0] for x in v])
        star="  <- ours" if tag=="joint_docclean" else ("  <- pretraining ablation" if tag=="joint_randinit" else "")
        print(f"{tag.replace('joint_',''):<24}{n:>3}  {m:.4f}+-{s:.4f}  "
              f"{np.mean([x[1] for x in v]):.4f} {np.mean([x[2] for x in v]):.4f} "
              f"{np.mean([x[3] for x in v]):.4f}{star}")
    if g.get("joint_docclean") and g.get("joint_randinit"):
        a=np.mean([x[0] for x in g["joint_docclean"]]); b=np.mean([x[0] for x in g["joint_randinit"]])
        print(f"\n  PRETRAINING EFFECT: {100*(a-b):+.1f} LAS points")
    # Headline margin must be against the STRONGEST baseline at ITS OWN tuned LR, not
    # against a baseline handicapped by our shared recipe (see lr_control below).
    if g.get("joint_docclean"):
        a=np.mean([x[0] for x in g["joint_docclean"]])
        cands={k:np.mean([x[0] for x in v]) for k,v in g.items()
               if k not in ("joint_docclean","joint_docclean_long","joint_randinit",
                            "joint_randinit_long") and len(v)>=8}
        if cands:
            bk=max(cands,key=cands.get)
            print(f"  vs strongest baseline ({bk.replace('joint_','')}, n>=8): "
                  f"{100*(a-cands[bk]):+.1f} LAS points")
    if g.get("joint_docclean") and g.get("joint_randinit_long"):
        a=np.mean([x[0] for x in g["joint_docclean"]])
        b=np.mean([x[0] for x in g["joint_randinit_long"]])
        print(f"  PRETRAINING EFFECT (converged random-init): {100*(a-b):+.1f} LAS points")

def lr_control():
    print("\n" + "-"*78)
    print("BASELINE-TUNING CONTROL (fold 0 / seed 0): is the shared recipe unfair to the")
    print("subword baselines? lr_enc swept for them AND for us.")
    print("-"*78)
    S=f"{GCB}/parser_data/runs"
    for enc,label in (("docclean","ours"),("philberta","PhilBerta"),("greberta","GreBerta")):
        row=[]
        b=os.path.join(S,f"joint_{enc}_f0_s0","test_scores_greedy.json")
        if os.path.exists(b): row.append(("1e-04*", json.load(open(b))["las"]))
        for d in sorted(glob.glob(f"{S}/joint_tune_{enc}_lr*_f0_s0")):
            q=os.path.join(d,"test_scores_greedy.json")
            if not os.path.exists(q): continue
            v=re.search(r"_lr([0-9.e-]+)_f0",os.path.basename(d)).group(1)
            row.append((f"{float(v):.0e}", json.load(open(q))["las"]))
        if not row: continue
        row.sort(key=lambda r: float(r[0].rstrip("*")))
        best=max(row,key=lambda r:r[1])
        print(f"  {label:<10} " + "  ".join(f"{k}:{v:.4f}" for k,v in row)
              + f"   -> best {best[0]} ({best[1]:.4f})")
    print("  (* = shared recipe used for the main table)")


def meter():
    print("\n"+"="*78); print("EXPERIMENT 3 -- METER (macronization, Norma test, git source, n=2660)"); print("="*78)
    rows=collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{REPO}/logs/eval-*.out")):
        txt=open(f, errors="ignore").read()
        if "--norma" not in txt: continue
        mm=re.search(r"model=(\S+)", txt); mt=re.search(r"macron test: (\{.*?\})", txt)
        if not (mm and mt): continue
        run=mm.group(1).split("/runs/")[-1].split("/")[0]
        if not run.startswith("meter_mac_v2"): continue
        d=json.loads(mt.group(1))
        rows["random-init" if "randinit" in run else "pretrained"][run]=(
            d["bal_acc"], d["acc"], d["macro_bal_acc"])
    if not rows: print("  (no results yet)"); return
    print(f"{'arm':<14}{'n':>3}  {'balanced acc':>16} {'acc':>8} {'macro-bal':>10}")
    means={}
    for arm in ("pretrained","random-init"):
        v=list(rows.get(arm,{}).values())
        if not v: continue
        m,s,n=_agg([x[0] for x in v]); means[arm]=m
        print(f"{arm:<14}{n:>3}  {m:.4f}+-{s:.4f}  {np.mean([x[1] for x in v]):.4f} "
              f"{np.mean([x[2] for x in v]):.4f}")
    if len(means)==2:
        print(f"\n  PRETRAINING EFFECT: {100*(means['pretrained']-means['random-init']):+.1f} points balanced accuracy")

def documentary():
    print("\n"+"="*78); print("EXPERIMENT 1 -- DOCUMENTARY RECONSTRUCTION (inscriptions + papyri)"); print("="*78)
    prot={"iphi_whole_unk":"inscriptions, whole-doc (no meta)",
          "iphi_clean_unk":"inscriptions, clean ctx (no meta)",
          "iphi_whole_meta":"inscriptions, whole-doc (+meta)",
          "pap_whole_unk":"papyri, whole-doc (no meta)"}
    for ver,label in (("v4","v4 (6-tier mix, pilot)"),
                      ("v3","v3 (pretrained)"),
                      ("v3_randinit","v3 RANDOM-INIT (pretraining ablation)"),
                      ("v2","v2")):
        agg=collections.defaultdict(list)
        for f in glob.glob(f"{REPO}/.scratch/evals/{ver}_t*.json"):
            p=os.path.basename(f)[len(ver)+1:-5][5:]
            if p not in prot: continue
            d=json.load(open(f)); agg[p].append((d["avg"]["CER"], d["avg"]["top1"], d["avg"]["top20"]))
        if not agg: continue
        print(f"\n  -- {label} --")
        for k,lab in prot.items():
            if k not in agg: continue
            m,s,n=_agg([x[0] for x in agg[k]])
            print(f"    {lab:<36} n={n:>2}  CER={m:.4f}+-{s:.4f}  "
                  f"top1={np.mean([x[1] for x in agg[k]]):.4f}  top20={np.mean([x[2] for x in agg[k]]):.4f}")
    print("\n  -- STRICT Ithaca-comparable (PHI digit-3 test, whole inscriptions,")
    print("     real lacunae in context, SPACES COUNTED, beam 20, no metadata) --")
    def merge(files):
        a=collections.defaultdict(lambda: dict(n=0,cer=0.0,cl=0.0,t1=0.0,t20=0.0))
        for f in files:
            try: d=json.load(open(f))
            except Exception: continue
            for r in d.get("per_L",[]):
                x=a[r["L"]]; n=r["n"]; x["n"]+=n; x["cer"]+=r["CER"]*n
                x["cl"]+=(r.get("CER_letters") or 0)*n; x["t1"]+=r["top1"]*n; x["t20"]+=r["top20"]*n
        return a
    for tag,pat in (("Ithaca (DeepMind release)", f"{INSC}/strict_f3/ithaca_shard*.json"),
                    ("ours old (both_ft_docclean)", f"{INSC}/strict_f3/ours_old.json"),
                    ("ours v2 (undertrained)", f"{INSC}/strict_f3/ours_v2_shard*.json"),
                    ("ours v3", f"{INSC}/strict_f3/ours_v3_shard*.json"),
                    ("ours v4 (pilot)", f"{INSC}/strict_f3/ours_v4_shard*.json")):
        fs=sorted(glob.glob(pat))
        if not fs: continue
        a=merge(fs)
        if not a: continue
        Ls=sorted(a)
        cer=np.mean([a[L]["cer"]/max(a[L]["n"],1) for L in Ls])
        cl=np.mean([a[L]["cl"]/max(a[L]["n"],1) for L in Ls])
        t1=np.mean([a[L]["t1"]/max(a[L]["n"],1) for L in Ls])
        t20=np.mean([a[L]["t20"]/max(a[L]["n"],1) for L in Ls])
        ntot=sum(a[L]["n"] for L in Ls)
        extra=f"  CER_let={cl:.4f}" if cl>0 else ""
        print(f"    {tag:<30} CER={cer:.4f}{extra}  top1={t1:.4f}  top20={t20:.4f}  (n={ntot})")

def scanner():
    print("\n" + "="*78)
    print("EXPERIMENT 3b -- SCANNER (Norma syllabify test, joint model, git source)")
    print("="*78)
    rows=collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{REPO}/logs/eval-*.out")):
        t=open(f,errors="ignore").read()
        mm=re.search(r"model=(\S+)",t); ms=re.search(r"norma syllabify \(test\) ===\n  (\{.*?\})",t)
        if not(mm and ms): continue
        run=mm.group(1).split("/runs/")[-1].split("/")[0]
        # joint_pilot is an older, different config -- not part of the seed-matched ablation
        if not run.startswith("meter_joint") or run=="meter_joint_pilot": continue
        d=json.loads(ms.group(1))
        arm="random-init" if "randinit" in run else "pretrained"
        rows[arm][run]=(d["bal_acc"], d["acc"], d["boundary_f1"])
    if not rows: print("  (no results yet)"); return
    print(f"{'arm':<14}{'n':>3}  {'balanced acc':>16} {'acc':>8} {'boundary F1':>12}")
    means={}
    for arm in ("pretrained","random-init"):
        v=list(rows.get(arm,{}).values())
        if not v: continue
        m,sd,n=_agg([x[0] for x in v]); means[arm]=m
        print(f"{arm:<14}{n:>3}  {m:.4f}+-{sd:.4f}  {np.mean([x[1] for x in v]):.4f} "
              f"{np.mean([x[2] for x in v]):.4f}")
    if len(means)==2:
        print(f"\n  PRETRAINING EFFECT: {100*(means['pretrained']-means['random-init']):+.1f} pts balanced accuracy")


if __name__=="__main__":
    w=sys.argv[1] if len(sys.argv)>1 else "all"
    if w in ("all","documentary"): documentary()
    if w in ("all","parsing"): parsing(); lr_control()
    if w in ("all","meter"): meter(); scanner()
    print()
