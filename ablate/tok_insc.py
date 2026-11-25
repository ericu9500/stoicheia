"""Documentary restoration for the flat-character ablation arm: fine-tune + strict evaluation.

Fine-tune: the same whole-inscription records as insc/train/finetune_whole.py (iphi only; real
lacunae kept in place as unsupervised MASK context), rendered to flat text and tokenized; the
same restoration noise mixture as the plane arms' restoration configs; fixed WSD schedule;
best checkpoint by greedy dev span CER on held-out-digit validation inscriptions.

Strict eval: reads the SAME frozen sample files as insc/eval/restore_strict.py (Ithaca char space,
spaces count, word division predicted). The flat model has one token per character of that
space ("▁" = space), so a gap of L characters is L mask slots; the beam is the same beam-20
non-sequential mask-predict as insc/eval/restore.py::beam_restore, over the 24 letters, final
sigma and the space token; hypotheses are ranked by length-normalized log-probability.

  python -m ablate.tok_insc finetune --config cfg.json
  python -m ablate.tok_insc strict --ckpt best.pt --samples s.json --out o.json [--shard i,k]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "insc" / "data"))
from ablate.render import render
from ablate.tok_model import TokConfig, TokenEncoder, load_token_encoder
from ablate.tok_noise import MASK, PAD, TokNoiseConfig, noise_tokens
from ablate.tok_train import load_tok, tok_tables, tok_loss
from train.schedule import wsd_dyn
from train.train import ddp_setup, save_ckpt

ALPHA = "αβγδεζηθικλμνξοπρστυφχψω"


def record_tokens(rec, tok):
    """plane record (with is_real_lacuna) -> (token ids, word_start, protect, letter->token)."""
    chars = np.asarray(rec["chars"], np.int64)
    lac = np.asarray(rec["is_real_lacuna"], bool)
    letters = np.where(chars < 24, chars, 0)
    bnd = np.asarray(rec["boundary"], np.int64); bnd = np.where(bnd >= 3, 0, bnd)
    dia = np.where(lac, 48, np.asarray(rec["dia"], np.int64))
    punct = np.where(lac, 0, np.asarray(rec["punct"], np.int64)); punct = np.where(punct >= 6, 0, punct)
    text, offs = render(letters, bnd, dia, np.where(lac, 0, rec["cap"]), punct)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = np.array(enc["input_ids"], np.int64)
    c2t = np.full(len(text) + 1, -1, np.int64)
    for ti, (a, e) in enumerate(enc["offset_mapping"]):
        c2t[a:max(e, a + 1)] = ti
    for k in range(len(c2t) - 2, -1, -1):
        if c2t[k] < 0:
            c2t[k] = c2t[k + 1]
    l2t = c2t[offs]
    protect = np.zeros(len(ids), bool); protect[l2t[lac]] = True
    return ids, l2t, protect


def finetune(cfg):
    os.environ["INSC_TEST_DIGIT"] = str(cfg["test_digit"]); os.environ["INSC_VAL_DIGIT"] = str(cfg["val_digit"])
    from insc.train.finetune_whole import load_domain_records
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    out = Path(os.path.expandvars(cfg["out_dir"])); out.mkdir(parents=True, exist_ok=True)
    torso = os.path.expandvars(cfg["torso"])
    random_init = torso.startswith("random:")
    enc, pcfg = load_token_encoder(torso.split(":", 1)[1] if random_init else torso, device,
                                   attn_impl=cfg.get("attn", "flex"), random_init=random_init)
    tok = load_tok(pcfg["tokenizer"])
    word_start, strip_map, sub_ids = tok_tables(tok, pcfg.get("dia_dropout", False))
    step0 = 0
    if (out / "last.pt").exists():
        sd = torch.load(out / "last.pt", map_location=device); enc.load_state_dict(sd["model"]); step0 = sd["step"]
    model = enc
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(enc, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": cfg.get("wd", 0.1)},
                             {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=device.type == "cuda")
    if step0 and "opt" in sd:
        opt.load_state_dict(sd["opt"])

    mix = {"iphi": 1.0, "papyri": 0.0}
    pools, _, _ = load_domain_records("train", cfg.get("min_len", 20), mix=mix, seq_len=cfg["seq_len_letters"])
    train_recs = [record_tokens(r, tok) for r in pools[0]]
    ncfg = TokNoiseConfig(**cfg.get("noise", {}))
    rng = np.random.default_rng(cfg["seed"] + step0 + 1000003 * rank)
    T, R, accum = cfg["seq_tokens"], cfg["micro_batch"], cfg.get("grad_accum", 1)

    def batch():
        ids = np.full((R, T), PAD, np.int64); lab = np.full((R, T), -100, np.int64)
        aux = np.full((R, T), -100, np.int64); lw = np.zeros((R, T), np.float32); seg = np.zeros((R, T), np.int64)
        for r in range(R):
            c = doc = misses = 0
            while c < T - 8:
                tids, _, prot = train_recs[int(rng.integers(len(train_recs)))]
                o = noise_tokens(tids, word_start[tids], ncfg, rng, sub_ids, strip_map, protect=prot)
                L = min(len(o["input_ids"]), T)
                if c + L > T:
                    if c == 0:
                        L = T
                    else:
                        misses += 1
                        if misses >= 6:
                            break
                        continue
                doc += 1; sl = slice(c, c + L)
                ids[r, sl] = o["input_ids"][:L]; lab[r, sl] = o["labels"][:L]; aux[r, sl] = o["aux_labels"][:L]
                lw[r, sl] = o["loss_w"]; seg[r, sl] = doc; c += L
        return {k: torch.from_numpy(v).to(device) for k, v in
                dict(input_ids=ids, labels=lab, aux_labels=aux, loss_w=lw, seg_id=seg).items()}

    fixture = None
    if rank == 0:
        vpools, _, _ = load_domain_records("val", cfg.get("min_len", 20), mix=mix, seq_len=cfg["seq_len_letters"])
        fixture = dev_fixture(vpools[0], tok, cfg.get("dev_cer_n_per_L", 60), cfg.get("dev_cer_L_max", 20), T)
    total, decay_len, warm = cfg["total_steps"], cfg.get("decay_len", 1500), cfg.get("warmup", 200)
    a0 = total - decay_len
    best, t0 = float("inf"), time.time()
    if (out / "best.json").exists():
        best = json.loads((out / "best.json").read_text())["dev_cer"]
    model.train()
    for step in range(step0, total):
        lr = cfg["lr"] * (step / warm if step < warm else
                          (1.0 if step < a0 else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - a0) / decay_len))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            b = batch()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, logs = tok_loss(fwd(b)["tok"], b, cfg.get("lam", 0.1))
            (loss / accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        if rank == 0 and step % cfg.get("log_every", 20) == 0:
            print(f"  step={step} lr={lr:.2e} gnorm={float(gn):.3f} " + " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                  + f" s/step={(time.time()-t0)/max(step-step0+1,1):.2f}", flush=True)
        if step > step0 and (step % cfg.get("eval_every", 250) == 0 or step == total - 1):
            if rank == 0:
                save_ckpt(dict(model=enc.state_dict(), opt=opt.state_dict(), step=step + 1, cfg=pcfg, ft_cfg=cfg), out / "last.pt")
                m = dev_span_cer(enc, fixture, device); m["step"] = step
                print("  DEV " + json.dumps(m), flush=True)
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
                if m["cer"] <= best:
                    best = m["cer"]
                    save_ckpt(dict(model=enc.state_dict(), step=step + 1, cfg=pcfg, ft_cfg=cfg), out / "best.pt")
                    (out / "best.json").write_text(json.dumps(dict(dev_cer=best, step=step)))
                    print(f"  new best.pt (dev CER {best:.4f})", flush=True)
                model.train()
            if is_ddp:
                import torch.distributed as dist
                dist.barrier()
    if rank == 0:
        print("FINETUNE DONE", flush=True)
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


def dev_fixture(recs, tok, n_per_L, L_max, T, seed=1234):
    """Mirror of finetune_whole.build_dev_cer_fixture: one letter span per doc (never on a real
    lacuna), whole doc as context; the gap covers the span's letter tokens and anything between them."""
    rng = np.random.default_rng(seed)
    out = []
    for L in range(1, L_max + 1):
        cands = [r for r in recs if L + 8 < len(r["chars"])]
        rng.shuffle(cands)
        made = 0
        for r in cands:
            if made >= n_per_L:
                break
            ids, l2t, prot = record_tokens(r, tok)
            if len(ids) > T:
                continue
            known = ~np.asarray(r["is_real_lacuna"], bool)
            valid = [s for s in range(4, len(r["chars"]) - L - 4) if known[s:s + L].all()]
            if not valid:
                continue
            s = int(rng.choice(valid))
            a, e = int(l2t[s]), int(l2t[s + L - 1]) + 1
            gold_letters = np.asarray(r["chars"][s:s + L], np.int64)
            out.append((ids, prot, a, e, l2t[s:s + L] - a, gold_letters))
            made += 1
    return out


def _base_letter_table(tok):
    """token id -> base letter id (0..23) or -1."""
    import unicodedata
    V = len(tok); inv = {i: s for s, i in tok.get_vocab().items()}
    tab = np.full(V, -1, np.int64)
    for i in range(V):
        s = inv.get(i, "")
        if len(s) == 1:
            b = unicodedata.normalize("NFD", s)[0].lower().replace("ς", "σ").replace("ϲ", "σ")
            if b in ALPHA:
                tab[i] = ALPHA.index(b)
    return tab


@torch.no_grad()
def dev_span_cer(enc, fixture, device, micro=16):
    from insc.train.finetune_whole import _levenshtein
    enc.eval()
    cers, exact = [], []
    lt = torch.tensor(_LETTER_TAB, device=device)
    for i in range(0, len(fixture), micro):
        chunk = fixture[i:i + micro]
        T = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), T), PAD, dtype=torch.long); seg = torch.zeros(len(chunk), T, dtype=torch.long)
        for b, (tids, prot, a, e, rel, gold) in enumerate(chunk):
            x = np.where(prot, MASK, tids).copy(); x[a:e] = MASK
            ids[b, :len(x)] = torch.from_numpy(x); seg[b, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = enc(dict(input_ids=ids.to(device), seg_id=seg.to(device)))["tok"].float()
        # restrict to letter tokens, then read the base letter of the argmax
        masked = lg.masked_fill((lt < 0)[None, None], float("-inf"))
        pred = lt[masked.argmax(-1)].cpu().numpy()
        for b, (tids, prot, a, e, rel, gold) in enumerate(chunk):
            p = pred[b, a + rel]
            cers.append(_levenshtein(list(p), list(gold)) / len(gold)); exact.append(int((p == gold).all()))
    enc.train()
    return dict(cer=round(float(np.mean(cers)), 4), exact=round(float(np.mean(exact)), 4), n=len(cers))


_LETTER_TAB = None   # token id -> base letter id, set in main() before fine-tuning


# ------------------------------------------------------------------ strict evaluation

@torch.no_grad()
def strict(ckpt, samples_path, out_path, beam_width=20, expand=48, shard=None):
    from insc.eval.restore_strict import canon, levenshtein, _write_report
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(os.path.expandvars(ckpt), map_location="cpu", weights_only=False)
    c = sd["cfg"]
    enc = TokenEncoder(TokConfig(vocab=c["vocab"], d_model=c["d_model"], n_heads=c["d_model"] // 64,
                                 depth=c["depth"], window=c["window"], attn_impl="sdpa")).to(device)
    enc.load_state_dict(sd["model"]); enc.eval()
    tok = load_tok(c["tokenizer"]); vocab = tok.get_vocab()
    SP = vocab["▁"]
    allowed = [vocab[ch] for ch in ALPHA] + [vocab["ς"], SP]
    render_tok = {vocab[ch]: ch for ch in ALPHA}; render_tok[vocab["ς"]] = "σ"; render_tok[SP] = " "
    allowed_t = torch.tensor(allowed, device=device)
    samples = json.loads(Path(os.path.expandvars(samples_path)).read_text())
    if shard:
        i, k = (int(x) for x in shard.split(",")); samples = samples[i::k]
    rows, per_sample, t0 = {}, [], time.time()
    unk = tok.unk_token_id

    def to_ids(text):
        seq = [SP]
        for ch in text:
            seq.append(SP if ch == " " else (MASK if ch == "-" else vocab.get(ch, unk)))
        return seq

    for si, smp in enumerate(samples):
        t, s, L, gold = smp["text"], smp["start"], smp["L"], canon(smp["gold"])
        base = np.array(to_ids(t), np.int64)
        gap = list(range(s + 1, s + 1 + L))                 # +1: leading "▁"
        base[gap] = MASK
        beam = [(base, tuple(gap), 0.0)]
        finished = {}
        while beam:
            x = torch.tensor(np.stack([h[0] for h in beam]), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                lg = enc(dict(input_ids=x, seg_id=torch.ones_like(x)))["tok"].float()
            lp = torch.log_softmax(lg, -1)[:, :, allowed_t]          # (B,T,A)
            children = {}
            for bi, (seq, rem, score) in enumerate(beam):
                rem = list(rem)
                sub = lp[bi, rem].clone()                             # (r, A)
                for ri, pos in enumerate(rem):                        # no double space
                    if seq[pos - 1] == SP or (pos + 1 < len(seq) and seq[pos + 1] == SP):
                        sub[ri, -1] = float("-inf")
                flat = sub.reshape(-1)
                top = torch.topk(flat, min(expand, flat.numel())).indices.cpu().numpy()
                for tt in top:
                    ri, ai = divmod(int(tt), len(allowed))
                    v = float(sub[ri, ai])
                    if not math.isfinite(v):
                        continue
                    pos = rem[ri]; new = seq.copy(); new[pos] = allowed[ai]
                    nrem = tuple(p for p in rem if p != pos); ns = score + v
                    if not nrem:
                        st = "".join(render_tok[int(q)] for q in new[gap])
                        if st.strip(" ") == "":
                            continue
                        if st not in finished or finished[st] < ns:
                            finished[st] = ns
                    else:
                        key = (new.tobytes(), nrem)
                        if key not in children or children[key][2] < ns:
                            children[key] = (new, nrem, ns)
            beam = sorted(children.values(), key=lambda h: -h[2])[:beam_width]
        ranked = sorted(((k, v / L) for k, v in finished.items()), key=lambda kv: -kv[1])[:beam_width]
        preds = [k for k, _ in ranked]
        pred = preds[0] if preds else ""
        r = rows.setdefault(L, dict(L=L, n=0, cers=[], t1=0, t20=0, cers_let=[]))
        r["cers"].append(levenshtein(pred, gold) / max(len(gold), 1))
        gl = gold.replace(" ", "")
        r["cers_let"].append(levenshtein(pred.replace(" ", ""), gl) / max(len(gl), 1))
        r["t1"] += int(pred == gold); r["t20"] += int(gold in preds); r["n"] += 1
        per_sample.append(dict(phi_id=smp["phi_id"], L=L, start=s, gold=gold, preds=preds,
                               scores=[round(v, 5) for _, v in ranked]))
        if (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(samples)} ({(time.time()-t0)/(si+1):.2f}s/sample)", flush=True)
            _write_report(rows, ckpt, out_path, si + 1, len(samples), per_sample=per_sample)
    _write_report(rows, ckpt, out_path, len(samples), len(samples), final=True, per_sample=per_sample)


def main():
    global _LETTER_TAB
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["finetune", "strict"])
    ap.add_argument("--config"); ap.add_argument("--ckpt"); ap.add_argument("--samples"); ap.add_argument("--out")
    ap.add_argument("--shard"); ap.add_argument("--beam", type=int, default=20)
    a = ap.parse_args()
    if a.mode == "finetune":
        cfg = json.loads(Path(a.config).read_text())
        torso = os.path.expandvars(cfg["torso"]).split(":")[-1]
        c = torch.load(torso, map_location="cpu", weights_only=False)["cfg"]
        _LETTER_TAB = _base_letter_table(load_tok(c["tokenizer"]))
        finetune(cfg)
    else:
        strict(a.ckpt, a.samples, a.out, beam_width=a.beam, shard=a.shard)


if __name__ == "__main__":
    main()
