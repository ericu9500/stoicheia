"""Paired significance tests for ALL three experiments.

Everything is paired on the unit that was actually replicated -- (fold, seed) for parsing,
fold for documentary, seed for meter -- so each comparison controls for split difficulty
and only the system varies. The headline p-value is an EXACT sign-flip permutation test
(no distributional assumption); note its floor is 2/2^n, so p=0.002 at n=10 is the
smallest attainable value, not a coincidence. d_z is paired Cohen's d.
"""
from __future__ import annotations
import collections, glob, json, math, os, re, sys
import numpy as np

GCB=os.path.expandvars("$STOICHEIA_DATA")
REPO=os.path.expandvars("$STOICHEIA_ROOT")

def _ttest_rel_p(d):
    """Two-sided paired t-test p-value without scipy (survival fn of |t| via a
    continued-fraction incomplete beta -- keeps this script dependency-free so the
    released code needs only numpy)."""
    n=len(d); sd=d.std(ddof=1)
    if sd==0 or n<2: return float("nan")
    t=abs(d.mean())/(sd/np.sqrt(n)); v=n-1
    x=v/(v+t*t)
    # regularised incomplete beta I_x(v/2, 1/2) via Lentz continued fraction
    a,b=v/2.0,0.5
    lbeta=(math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b))
    def betacf(a,b,x,itmax=300,eps=3e-16):
        qab,qap,qam=a+b,a+1.0,a-1.0
        c=1.0; d_=1.0-qab*x/qap
        if abs(d_)<1e-30: d_=1e-30
        d_=1.0/d_; h=d_
        for m in range(1,itmax+1):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2))
            d_=1.0+aa*d_;  c=1.0+aa/c
            if abs(d_)<1e-30: d_=1e-30
            if abs(c)<1e-30: c=1e-30
            d_=1.0/d_; h*=d_*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d_=1.0+aa*d_;  c=1.0+aa/c
            if abs(d_)<1e-30: d_=1e-30
            if abs(c)<1e-30: c=1e-30
            d_=1.0/d_; delta=d_*c; h*=delta
            if abs(delta-1.0)<eps: break
        return h
    if x < (a+1.0)/(a+b+2.0):
        ib=np.exp(a*np.log(x)+b*np.log(1-x)-lbeta)*betacf(a,b,x)/a
    else:
        ib=1.0-np.exp(b*np.log(1-x)+a*np.log(x)-lbeta)*betacf(b,a,1-x)/b
    return float(min(1.0, max(0.0, ib)))


def perm_p(diff):
    n=len(diff); obs=abs(diff.mean())
    if n<=20:
        c=t=0
        for m in range(1<<n):
            s=np.array([1 if m>>i&1 else -1 for i in range(n)])
            c+= abs((diff*s).mean())>=obs-1e-12; t+=1
        return c/t
    rng=np.random.default_rng(0)
    s=rng.choice([-1,1],size=(100000,n))
    return float((np.abs((diff*s).mean(1))>=obs-1e-12).mean())

def report(title, unit, a_name, a, b_name, b, higher_better=True):
    keys=sorted(set(a)&set(b))
    if len(keys)<3:
        print(f"  {a_name} vs {b_name}: only {len(keys)} paired {unit}s -- not testable yet"); return
    x=np.array([a[k] for k in keys]); y=np.array([b[k] for k in keys])
    d=(x-y) if higher_better else (y-x)
    p=perm_p(d); dz=d.mean()/(d.std(ddof=1)+1e-12)
    t=_ttest_rel_p(d)
    print(f"  {a_name} vs {b_name}: n={len(keys)} {unit}s  mean diff={d.mean():+.4f}  "
          f"perm p={p:.2e}  t p={t:.1e}  d_z={dz:.1f}")

def parsing():
    print("\nPARSING (LAS, paired by fold+seed; higher better)")
    S=f"{GCB}/parser_data/runs"; g=collections.defaultdict(dict)
    for d in sorted(glob.glob(f"{S}/joint_*_f*_s*")):
        f=os.path.join(d,"test_scores_greedy.json")
        if not os.path.exists(f): continue
        j=json.load(open(f))
        if j.get("las") is None: continue
        b=os.path.basename(d); m=re.search(r"_(f\d+_s\d+)$",b)
        g[re.sub(r"_f\d+_s\d+$","",b)][m.group(1)]=j["las"]
    ref=g.get("joint_docclean",{})
    for k in sorted(g):
        if k=="joint_docclean": continue
        # joint_tune_* are single-fold LR-sweep diagnostics (reported separately as the
        # baseline-tuning control), not systems to significance-test.
        if k.startswith("joint_tune_"): continue
        report("", "run", "ours", ref, k.replace("joint_",""), g[k])

def documentary():
    print("\nDOCUMENTARY (CER, paired by fold; LOWER better)")
    def load(ver):
        out={}
        for f in glob.glob(f"{REPO}/.scratch/evals/{ver}_t*_iphi_whole_unk.json"):
            tv=re.search(r"_(t\dv\d)_",os.path.basename(f)).group(1)
            out[tv]=json.load(open(f))["avg"]["CER"]
        return out
    v3=load("v3"); ri=load("v3_randinit"); v2=load("v2")
    if v3 and ri: report("", "fold", "v3", v3, "random-init", ri, higher_better=False)
    if v3 and v2: report("", "fold", "v3", v3, "v2 (superseded)", v2, higher_better=False)
    if not v3: print("  (v3 evals not written yet)")

def strict_ithaca():
    print("\nSTRICT vs ITHACA (top-1/top-20 exact-match, two-proportion z-test, n=3000 each side)")
    print("  (samples are IDENTICAL on both sides -- same frozen file -- so this unpaired")
    print("   test is conservative; a paired McNemar test would only be stronger)")
    INSC=os.path.expandvars("$STOICHEIA_DATA")
    import collections
    def merge(pat):
        a=collections.defaultdict(lambda: dict(n=0,t1=0.0,t20=0.0))
        for f in glob.glob(pat):
            try: d=json.load(open(f))
            except Exception: continue
            for r in d.get("per_L",[]):
                x=a[r["L"]]; x["n"]+=r["n"]; x["t1"]+=r["top1"]*r["n"]; x["t20"]+=r["top20"]*r["n"]
        n=sum(v["n"] for v in a.values())
        return (n, sum(v["t1"] for v in a.values()), sum(v["t20"] for v in a.values())) if n else None
    def ztest(k1,n1,k2,n2):
        p1,p2=k1/n1,k2/n2; p=(k1+k2)/(n1+n2)
        se=math.sqrt(p*(1-p)*(1/n1+1/n2))
        z=(p1-p2)/se
        # two-sided normal sf via erfc
        pv=math.erfc(abs(z)/math.sqrt(2))
        return z,pv
    ith=merge(f"{INSC}/strict_f3/ithaca_shard*.json")
    if not ith: print("  (ithaca results missing)"); return
    for tag,pat in (("ours old",f"{INSC}/strict_f3/ours_old.json"),
                    ("ours v3",f"{INSC}/strict_f3/ours_v3_shard*.json"),
                    ("ours v4",f"{INSC}/strict_f3/ours_v4_shard*.json")):
        m=merge(pat)
        if not m: continue
        for name,ki,ko in (("top-1",1,1),("top-20",2,2)):
            z,pv=ztest(m[ki],m[0],ith[ko],ith[0])
            print(f"  {tag:<9} vs Ithaca {name:<7}: {m[ki]/m[0]:.4f} vs {ith[ko]/ith[0]:.4f}"
                  f"   z={z:+.2f}  p={pv:.2e}")


def meter():
    print("\nMETER (macron balanced acc, paired by seed; higher better)")
    rows=collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{REPO}/logs/eval-*.out")):
        txt=open(f,errors="ignore").read()
        if "--norma" not in txt: continue
        mm=re.search(r"model=(\S+)",txt); mt=re.search(r"macron test: (\{.*?\})",txt)
        if not(mm and mt): continue
        run=mm.group(1).split("/runs/")[-1].split("/")[0]
        if not run.startswith("meter_mac_v2"): continue
        seed=re.search(r"_s(\d)$",run); seed=seed.group(1) if seed else "0"
        arm="randinit" if "randinit" in run else "pretrained"
        rows[arm][seed]=json.loads(mt.group(1))["bal_acc"]
    if rows.get("pretrained") and rows.get("randinit"):
        report("", "seed", "pretrained", rows["pretrained"], "random-init", rows["randinit"])
    else: print("  (not enough scored runs yet)")

if __name__=="__main__":
    w=sys.argv[1] if len(sys.argv)>1 else "all"
    if w in ("all","parsing"): parsing()
    if w in ("all","documentary"): documentary(); strict_ithaca()
    if w in ("all","meter"): meter()
    print()
