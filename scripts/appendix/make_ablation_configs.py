"""Write every pretraining + downstream config of the ablation program.

All pretraining arms: same documentary-free corpus (data_nodoc), same 4096-letter windows,
same rows/step, same optimizer/schedule/step budget, final checkpoint used downstream.

  python scripts/appendix/make_ablation_configs.py --steps 30000 --rows 32 --out configs/ablate
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

R = "$STOICHEIA_DATA"
RUNS = f"{R}/runs/ablate"
TOK = f"{R}/ablate/tok"
STAGED = [[0.5, [1.0, 1.0, 0.3]], [0.9, [1.0, 1.0, 0.0]], [1.0, [1.0, 0.0, 0.0]]]


def pretrain_arms(steps, rows):
    common = dict(seed=0, qk_norm=True, grad_accum=2, micro_batch=rows, lr=6e-4, wd=0.1, clip=1.0,
                  warmup_frac=0.04, decay_frac=0.2, lam=0.1, total_steps=steps, num_workers=12,
                  log_every=20, ckpt_every=1000, eval_n=128, compile=True, attn="flex")
    planes = dict(common, d_model=768, depth=12, char_window=256, seq_len=4096, auto_anneal=False,
                  eval_every_ckpt=True, tier_weights=[1.0, 1.0, 0.3], anneal_phases=STAGED)
    arms = {
        # E3 reference (2 seeds) and design ablations
        "planes_ref_s0": dict(planes),
        "planes_ref_s1": dict(planes, seed=1),
        "planes_mlm15": dict(planes, noise=dict(t_min=0.15, t_max=0.15)),
        "planes_iidnoise": dict(planes, noise=dict(w_span=0, w_word=0, w_elastic=0, w_iid=1.0,
                                                   w_halfword=0, w_substitute=0)),
        # E2 corpus tiers
        "data_gold": dict(planes, tier_weights=[1.0, 0.0, 0.0], anneal_phases=[[1.0, [1.0, 0.0, 0.0]]]),
        "data_silver": dict(planes, tier_weights=[0.0, 1.0, 0.0], anneal_phases=[[1.0, [0.0, 1.0, 0.0]]]),
        "data_gold_silver": dict(planes, tier_weights=[1.0, 1.0, 0.0],
                                 anneal_phases=[[0.9, [1.0, 1.0, 0.0]], [1.0, [1.0, 0.0, 0.0]]]),
        "data_gold_bronze": dict(planes, tier_weights=[1.0, 0.0, 0.3],
                                 anneal_phases=[[0.5, [1.0, 0.0, 0.3]], [1.0, [1.0, 0.0, 0.0]]]),
        # E4 size
        "planes_medium": dict(planes, d_model=1024, depth=16),
    }
    tok = dict(common, d_model=768, depth=12, window_letters=4096, tier_weights=[1.0, 1.0, 0.3],
               anneal_phases=STAGED, ckpt_every=1000)
    arms["tok_char"] = dict(tok, tokenizer=f"{TOK}/char", dia_dropout=True, window=320, seq_tokens=5376)
    arms["tok_bpe"] = dict(tok, tokenizer=f"{TOK}/bpe", dia_dropout=False, window=64, seq_tokens=1280)
    for k, v in arms.items():
        v["name"] = k; v["out_dir"] = f"{RUNS}/pretrain/{k}"
    return arms


def downstream(arm, ckpt, kind):
    """kind: planes | tok_char | tok_bpe"""
    out = {}
    base = json.loads(Path("configs/syntax/joint_docclean_f0_s0.json").read_text())
    seeds = [0, 1] if arm in ("planes_ref_s0",) else [0]
    for f in range(5):
        for s in seeds:
            c = dict(base, ckpt=ckpt, fold=f, seed=s, name=f"parse_{arm}_f{f}_s{s}",
                     kfold_dir=f"{R}/treebanks/oga_repo/kfold", micro_batch=16, eval_micro=32,
                     out_dir=f"{RUNS}/parse/{arm}_f{f}_s{s}")
            out[f"parse/{arm}_f{f}_s{s}"] = c
    mbase = json.loads(Path("configs/meter/mac_v2.json").read_text())
    for s in range(3):
        out[f"meter/{arm}_mac_s{s}"] = dict(mbase, ckpt=ckpt, seed=s, name=f"mac_{arm}_s{s}",
                                            encoded=f"{R}/data/meter/encoded", out_dir=f"{RUNS}/meter/{arm}_mac_s{s}")
    if kind == "planes":
        ibase = json.loads(Path("configs/insc/finetune_whole_v3_t3v4.json").read_text())
        pc = json.loads(Path(f"configs/ablate/pretrain/{arm if not arm.endswith('_rand') else 'planes_ref_s0'}.json").read_text())
        for s in range(2):
            out[f"insc/{arm}_t3_s{s}"] = dict(ibase, name=f"insc_{arm}_t3_s{s}", seed=s, d_model=pc["d_model"],
                                             depth=pc["depth"], torso=ckpt if not ckpt.startswith("random:") else "random",
                                             mix={"iphi": 1.0, "papyri": 0.0}, meta_condition=False, attn="flex",
                                             seq_len=4096, micro_batch=16, grad_accum=2, total_steps=6000,
                                             decay_len=1500, anneal_patience=0, fixed_anneal_start=4500,
                                             out_dir=f"{RUNS}/insc/{arm}_t3_s{s}")
    elif kind == "tok_char":
        ibase = json.loads(Path("configs/insc/finetune_whole_v3_t3v4.json").read_text())
        for s in range(2):
            out[f"insc/{arm}_t3_s{s}"] = dict(name=f"insc_{arm}_t3_s{s}", seed=s, torso=ckpt, test_digit="3", val_digit="4",
                                             noise={k: v for k, v in ibase["noise"].items() if not k.startswith("p_")},
                                             seq_len_letters=4096, seq_tokens=5376, micro_batch=16, grad_accum=2,
                                             lr=ibase["lr"], wd=0.1, warmup=200, total_steps=6000, decay_len=1500,
                                             eval_every=250, dev_cer_n_per_L=60, dev_cer_L_max=20, attn="flex",
                                             out_dir=f"{RUNS}/insc/{arm}_t3_s{s}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--out", default="configs/ablate")
    a = ap.parse_args()
    out = Path(a.out)
    arms = pretrain_arms(a.steps, a.rows)
    (out / "pretrain").mkdir(parents=True, exist_ok=True)
    for k, v in arms.items():
        (out / "pretrain" / f"{k}.json").write_text(json.dumps(v, indent=1))
    n = 0
    for k, v in arms.items():
        kind = "planes" if not k.startswith("tok_") else k
        for name, c in downstream(k, f"{v['out_dir']}/final.pt", kind).items():
            p = out / f"{name}.json"; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(c, indent=1)); n += 1
    # random-init floors at the ablation size (same architecture as planes_ref_s0)
    for name, c in downstream("planes_rand", f"random:{arms['planes_ref_s0']['out_dir']}/final.pt", "planes").items():
        p = out / f"{name}.json"; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(c, indent=1)); n += 1
    print(f"{len(arms)} pretraining arms, {n} downstream configs -> {out}")


if __name__ == "__main__":
    main()
