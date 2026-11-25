"""Pretraining loop for the token-level ablation arms (flat characters / BPE subwords).

Same records, windows, tier weights, staged anneal, optimizer, schedule, batch geometry, step
budget and transformer as the plane arms trained by train/train.py; differs only in the input
representation (ablate/tok_noise.py, ablate/tok_model.py). Fixed budget (no dev-driven stop),
so every ablation arm is trained for the same number of optimizer steps on the same text.

  torchrun --nproc_per_node=4 -m ablate.tok_train --config configs/ablate/tok_char.json
"""
from __future__ import annotations

import argparse, contextlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ablate.render import render, strip_diacritics
from ablate.tok_model import TokConfig, TokenEncoder
from ablate.tok_noise import MASK, PAD, TokNoiseConfig, noise_tokens
from train.data import HOLDOUT_MOD, MultiTierLoader, stable_cfg
from train.schedule import wsd_dyn
from train.train import ddp_setup, infinite_records, save_ckpt


def load_tok(tok_dir):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(os.path.expandvars(tok_dir))


def tok_tables(tok, strip: bool):
    V = len(tok)
    vocab = tok.get_vocab()
    inv = {i: s for s, i in vocab.items()}
    word_start = np.zeros(V, bool)
    strip_map = np.arange(V, dtype=np.int64)
    sub = []
    for i in range(V):
        s = inv.get(i, "")
        word_start[i] = s.startswith("▁")
        if i >= 4:
            sub.append(i)
        if strip and i >= 4 and len(s) == 1:
            j = vocab.get(strip_diacritics(s))
            if j is not None:
                strip_map[i] = j
    return word_start, (strip_map if strip else None), np.array(sub, np.int64)


class TokBatches(torch.utils.data.IterableDataset):
    def __init__(self, gdata, tier_weights, cfg, rows, seed, rank, world):
        super().__init__()
        self.__dict__.update(locals())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        gshard, gtot = self.rank * nw + wid, self.world * nw
        cfg = self.cfg
        tok = load_tok(cfg["tokenizer"])
        word_start, strip_map, sub_ids = tok_tables(tok, cfg.get("dia_dropout", False))
        dcfg = stable_cfg(self.gdata, w=tuple(self.tier_weights), window=cfg["window_letters"],
                          seed=self.seed)
        loader = MultiTierLoader(dcfg, rank=gshard, world_size=gtot)
        rng = np.random.default_rng(self.seed * 100003 + gshard)
        ncfg = TokNoiseConfig(**cfg.get("noise", {}))
        T, R = cfg["seq_tokens"], self.rows
        gen = infinite_records(loader)
        pending = None
        while True:
            ids = np.full((R, T), PAD, np.int64); lab = np.full((R, T), -100, np.int64)
            aux = np.full((R, T), -100, np.int64); lw = np.zeros((R, T), np.float32)
            seg = np.zeros((R, T), np.int64)
            for r in range(R):
                c, doc, misses = 0, 0, 0
                while c < T - 8:
                    if pending is None:
                        rec = next(gen)
                        text, _ = render(rec["chars"], rec["boundary"], rec["dia"], rec["cap"], rec["punct"])
                        tids = np.array(tok(text, add_special_tokens=False)["input_ids"], np.int64)
                        pending = noise_tokens(tids, word_start[tids], ncfg, rng, sub_ids, strip_map)
                    L = len(pending["input_ids"])
                    if L > T:
                        L = T
                    if c + L > T:
                        if c == 0:
                            L = T
                        else:
                            misses += 1
                            if misses >= 6:
                                break
                            pending = None           # drop: try a smaller record for the tail
                            continue
                    doc += 1
                    sl = slice(c, c + L)
                    ids[r, sl] = pending["input_ids"][:L]; lab[r, sl] = pending["labels"][:L]
                    aux[r, sl] = pending["aux_labels"][:L]; lw[r, sl] = pending["loss_w"]
                    seg[r, sl] = doc
                    c += L; pending = None
            yield dict(input_ids=torch.from_numpy(ids), labels=torch.from_numpy(lab),
                       aux_labels=torch.from_numpy(aux), loss_w=torch.from_numpy(lw),
                       seg_id=torch.from_numpy(seg))


def make_loader(gdata, w, cfg, rows, seed, rank, world):
    ds = TokBatches(gdata, list(w), cfg, rows, seed, rank, world)
    nwk = cfg.get("num_workers", 12)
    return iter(torch.utils.data.DataLoader(ds, batch_size=None, num_workers=nwk,
                                            prefetch_factor=4 if nwk else None,
                                            persistent_workers=nwk > 0, pin_memory=True))


def tok_loss(logits, batch, lam):
    lab = batch["labels"]; m = lab != -100
    ce = F.cross_entropy(logits[m].float(), lab[m], reduction="none")
    w = batch["loss_w"][m]
    loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
    logs = dict(tok=loss.item())
    am = batch["aux_labels"] != -100
    if am.any():
        a = F.cross_entropy(logits[am].float(), batch["aux_labels"][am])
        loss = loss + lam * a; logs["dia"] = a.item()
    else:
        loss = loss + logits.sum() * 0.0
    return loss, logs


@torch.no_grad()
def holdout_eval(model, cfg, gdata, device, n=256, rate=0.3):
    """nats/token on held-out pristine records (idx % HOLDOUT_MOD == 0), iid masking at t=0.3."""
    import pyarrow.parquet as pq
    d = Path(gdata) / "shards" / "v1_punct"
    ip = d / "index_dedup.parquet"
    idx = pq.read_table(ip if ip.exists() else d / "index.parquet")
    offs = idx.column("offset").to_numpy(); lens = idx.column("length").to_numpy()
    tier = idx.column("tier").to_numpy(zero_copy_only=False)
    el = np.flatnonzero((tier == "pristine") & (lens >= 64) & (lens <= 2048))
    el = el[el % HOLDOUT_MOD == 0][:n]
    planes = {p: np.memmap(d / f"{p}.bin", dtype=np.uint8, mode="r")
              for p in ("chars", "boundary", "dia", "cap", "punct")}
    tok = load_tok(cfg["tokenizer"])
    rng = np.random.default_rng(0)
    tot, cnt = 0.0, 0
    model.eval()
    for i in el:
        o, l = int(offs[i]), int(lens[i])
        text, _ = render(*(np.asarray(planes[p][o:o + l]) for p in ("chars", "boundary", "dia", "cap", "punct")))
        ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][:cfg["seq_tokens"]])
        m = torch.from_numpy(rng.random(len(ids)) < rate)
        if not m.any():
            continue
        inp = ids.clone(); inp[m] = MASK
        b = dict(input_ids=inp[None].to(device), seg_id=torch.ones(1, len(ids), dtype=torch.long, device=device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(b)["tok"][0]
        tot += F.cross_entropy(lg[m.to(device)].float(), ids[m].to(device), reduction="sum").item()
        cnt += int(m.sum())
    model.train()
    return dict(nats_per_masked_token=round(tot / max(cnt, 1), 4), n_tokens=cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    gdata = os.environ["STOICHEIA_DATA"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    tok = load_tok(cfg["tokenizer"])
    cfg["vocab"] = len(tok); cfg["arch"] = "tok"
    mcfg = TokConfig(vocab=cfg["vocab"], d_model=cfg["d_model"], n_heads=cfg["d_model"] // 64,
                     depth=cfg["depth"], window=cfg["window"], attn_impl=cfg.get("attn", "flex"),
                     qk_norm=cfg.get("qk_norm", True))
    model = TokenEncoder(mcfg).to(device)
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"params={n/1e6:.1f}M vocab={cfg['vocab']} T={cfg['seq_tokens']} world={world}", flush=True)
    core = model
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model
    decay = [p for p in model.parameters() if p.ndim >= 2]
    nodecay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]},
                             {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=device.type == "cuda")
    step0 = 0
    if (out / "last.pt").exists():
        sd = torch.load(out / "last.pt", map_location=device)
        core.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"]); step0 = sd["step"]
        if rank == 0:
            print(f"resumed from step {step0}", flush=True)
    total = cfg["total_steps"]
    if step0 >= total:
        if rank == 0:
            print("run already complete", flush=True)
        return
    a0 = int(total * (1 - cfg.get("decay_frac", 0.2)))
    decay_len = total - a0
    rows, accum = cfg["micro_batch"], cfg.get("grad_accum", 1)
    seed = cfg["seed"] + step0
    torch.manual_seed(cfg["seed"] + rank + step0 * 7919)
    phases = cfg.get("anneal_phases") or [[1.0, cfg["tier_weights"]]]
    its = {}

    def it_for(w):
        k = tuple(w)
        if k not in its:
            its[k] = make_loader(gdata, w, cfg, rows, seed, rank, world)
        return its[k]

    t0, seen, logf = time.time(), 0, out / f"metrics_rank{rank}.jsonl"
    model.train()
    for step in range(step0, total):
        if step < a0:
            w = cfg["tier_weights"]
        else:
            fr = (step - a0) / max(decay_len, 1)
            w = next(pw for f, pw in phases if fr < f or f == phases[-1][0])
        it = it_for(w)
        lr = cfg["lr"] * wsd_dyn(step, total, a0, cfg.get("warmup_frac", 0.04), cfg.get("decay_frac", 0.2))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
            ctx = model.no_sync() if (is_ddp and micro < accum - 1) else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = fwd(batch)["tok"]
                    loss, logs = tok_loss(logits, batch, cfg.get("lam", 0.1))
                (loss / accum).backward()
            seen += int((batch["seg_id"] > 0).sum()) * world
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        if torch.isfinite(gn):
            opt.step()
        if rank == 0 and step % cfg.get("log_every", 20) == 0:
            rec = dict(step=step, lr=round(lr, 6), gnorm=round(float(gn), 3),
                       tokps=round(seen / (time.time() - t0) / 1e6, 3),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else 0,
                       **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(logf, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if rank == 0 and step > step0 and step % cfg.get("ckpt_every", 1000) == 0:
            save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(), step=step + 1, cfg=cfg), out / "last.pt")
            try:
                m = holdout_eval(core, cfg, gdata, device, n=cfg.get("eval_n", 128)); m["step"] = step
                print("  EVAL " + json.dumps(m), flush=True)
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
            except Exception as e:
                print(f"  EVAL failed: {str(e)[:200]}", flush=True)
    if rank == 0:
        st = dict(model=core.state_dict(), opt=opt.state_dict(), step=total, cfg=cfg)
        save_ckpt(st, out / "last.pt")
        save_ckpt(dict(model=core.state_dict(), step=total, cfg=cfg), out / "final.pt")
        print(f"DONE step {total}", flush=True)
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
