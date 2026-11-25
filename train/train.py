"""Config-driven pretraining loop for Stoicheia.

Single-process or torchrun/DDP. One JSON config freezes size, schedule, and budget.
Two-phase curriculum: 3-tier stable phase -> gold-only anneal in the WSD decay window.
Checkpoint/resume by step (every ckpt_every steps) — safe to run as a chain of
independent, dependency-linked SLURM jobs instead of one long request.

  torchrun --nproc_per_node=4 -m train.train --config configs/stoicheia.json
"""
from __future__ import annotations

import argparse, contextlib, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.char_bert import CharBertConfig, CharBertEncoder, num_params
from train.collate import pack_batch
from train.data import MultiTierLoader, stable_cfg, anneal_cfg
from train.loss import compute_loss
from train.noising import NoiseConfig
from train.schedule import wsd_dyn


def ddp_setup():
    if "RANK" in os.environ:
        import datetime
        import torch.distributed as dist
        # Generous timeout: rank 0 writes a ~5GB checkpoint to shared disk and then
        # evaluates, while every other rank waits in the next collective; on a busy
        # shared filesystem that write can exceed 60 min. Do not bound the eval with
        # signal.alarm(): SIGALRM interrupts the write and torch.save fails with EINTR.
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=180))
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, True
    return 0, 1, False


def eval_stalled(eval_path: Path, window=8, eps=0.002, split="train"):
    """True if held-out bits_per_char has stopped improving: best of the most recent
    window/2 evals is not at least eps bits better than the best of the window/2 before.
    Eval masking is deterministic (fixed seed + records), so the curve is low-noise and a
    small eps suffices. Only entries of the given eval split are considered, so a run
    that switched eval source (train-holdout -> dev) restarts its stall window cleanly."""
    if not eval_path.exists():
        return False
    bpc = []
    for line in eval_path.read_text().splitlines():
        try:
            e = json.loads(line)
            if e.get("split", "train") == split:
                bpc.append(e["bits_per_char"])
        except Exception:
            continue
    if len(bpc) < window:
        return False
    half = window // 2
    old_best = min(bpc[-window:-half])
    new_best = min(bpc[-half:])
    return new_best > old_best - eps


def save_ckpt(obj, path: Path):
    """Atomic checkpoint write: a wall-time kill mid-save must never corrupt the resume
    point (the chain of jobs depends on last.pt always being loadable)."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def infinite_records(loader, chunk=256):
    while True:
        for r in loader.records(chunk):
            yield r


class BatchDataset(torch.utils.data.IterableDataset):
    """Produces packed batches under a multiprocess DataLoader — N worker processes build
    batches in PARALLEL. Single-threaded collate (masking is a per-record Python loop) is
    slow enough to starve the GPU otherwise. Each (rank, worker) gets a disjoint data shard
    and its own RNG, so ranks/workers never collide and runs stay reproducible."""
    def __init__(self, gdata, tier_weights, gold_only, ncfg, T, rows, seed, rank, world,
                 exclude_holdout=True):
        super().__init__()
        self.__dict__.update(locals())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gshard = self.rank * nw + wid
        gtot = self.world * nw
        cfg = (anneal_cfg(self.gdata, window=self.T, seed=self.seed,
                          exclude_holdout=self.exclude_holdout)
               if self.gold_only else
               stable_cfg(self.gdata, w=tuple(self.tier_weights), window=self.T, seed=self.seed,
                          exclude_holdout=self.exclude_holdout))
        loader = MultiTierLoader(cfg, rank=gshard, world_size=gtot)
        g = torch.Generator().manual_seed(self.seed * 100003 + gshard)
        gen = infinite_records(loader)
        while True:
            yield pack_batch(gen, self.ncfg, self.T, self.rows, g)


def make_loader(gdata, tier_weights, gold_only, ncfg, T, rows, seed, rank, world, num_workers,
                exclude_holdout=True):
    ds = BatchDataset(gdata, tier_weights, gold_only, ncfg, T, rows, seed, rank, world,
                      exclude_holdout=exclude_holdout)
    return torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        prefetch_factor=(2 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0), pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])   # allow "$STOICHEIA_DATA/..." in configs
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gdata = os.environ["STOICHEIA_DATA"]
    # dev-driven regime: eval_shards points at the fold's REAL val split (unseen works);
    # all training decisions (anneal stall, early stop, best.pt) then key off dev, and
    # train_holdout=false returns the intra-train mod-200 holdout to training.
    eval_shards = cfg.get("eval_shards")
    if eval_shards:
        eval_shards = os.path.expandvars(eval_shards)
    eval_split = "val" if eval_shards else "train"
    T = cfg["seq_len"]; rows = cfg.get("rows", cfg["micro_batch"])
    total = cfg["total_steps"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    metrics_f = out / f"metrics_rank{rank}.jsonl"

    mcfg = CharBertConfig(attn_impl=cfg.get("attn", "flex"), d_model=cfg["d_model"],
                          n_heads=cfg["d_model"] // 64, depth=cfg["depth"],
                          char_window=cfg["char_window"], qk_norm=cfg.get("qk_norm", True))
    model = CharBertEncoder(mcfg).to(device)
    if rank == 0:
        print(f"params={num_params(model)/1e6:.1f}M attn={mcfg.attn_impl} "
              f"T={T} rows={rows} world={world} total_steps={total}")
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model

    # decay only matrices; 1-D params (RMSNorm gains) are shape-constrained, not capacity
    decay_p = [p for p in model.parameters() if p.ndim >= 2]
    nodecay_p = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": cfg["wd"]},
         {"params": nodecay_p, "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"))
    # tuned defaults: span/word/elastic/iid/halfword/substitute; a config "noise" dict overrides
    # individual fields (objective/corruption ablations: fixed-rate MLM, iid-only corruption)
    ncfg = NoiseConfig(**cfg.get("noise", {}))

    w = tuple(cfg.get("tier_weights", [1.0, 1.0, 0.3]))
    nworkers = cfg.get("num_workers", 16)

    step0 = 0
    ckpt = out / "last.pt"
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        (model.module if is_ddp else model).load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"]); step0 = sd["step"]
        if rank == 0:
            print(f"resumed from step {step0}")

    # completion check must mirror the dynamic schedule below: the run ends at
    # anneal_start + decay_len (marker-decided or capped), not at nominal total_steps.
    _decay_len = total - int(total * (1 - cfg.get("decay_frac", 0.2)))
    _hard_max = max(cfg.get("hard_max_steps", total), total)
    _marker = out / "anneal_start.json"
    if cfg.get("auto_anneal", True) and _marker.exists():
        _end = min(json.loads(_marker.read_text())["step"], _hard_max - _decay_len) + _decay_len
    elif cfg.get("auto_anneal", True):
        _end = _hard_max
    else:
        _end = total
    if step0 >= _end:
        if rank == 0:
            print(f"run already complete (resume step {step0} >= end step {_end}); "
                  f"nothing to do")
        if is_ddp:
            import torch.distributed as dist
            dist.destroy_process_group()
        return

    # fold the resume step into every RNG so each chained job sees a FRESH data order and
    # fresh noise masks instead of replaying the stream from position 0 (the loader is an
    # infinite sampler — there is no cheap "skip to batch step0", a reseed is equivalent).
    data_seed = cfg["seed"] + step0
    torch.manual_seed(cfg["seed"] + rank + step0 * 7919)   # drives DataLoader worker seeds

    excl_holdout = cfg.get("train_holdout", True)
    stable_it = iter(make_loader(gdata, list(w), False, ncfg, T, rows, data_seed, rank, world,
                                 nworkers, exclude_holdout=excl_holdout))
    # anneal data: default = classic gold-only switch. anneal_phases staggers the mix over
    # the decay window instead — [[end_frac, [w_gold, w_silver, w_bronze]], ...] — so the
    # highest-repetition data only dominates once the LR is too small to overfit on it.
    phases = cfg.get("anneal_phases")
    if phases:
        phase_ends, phase_its, _cache = [], [], {tuple(w): stable_it}
        for f, pw in phases:
            key = tuple(pw)
            if key not in _cache:
                _cache[key] = iter(make_loader(gdata, list(pw), False, ncfg, T, rows,
                                               data_seed, rank, world, nworkers,
                                               exclude_holdout=excl_holdout))
            phase_ends.append(f); phase_its.append(_cache[key])
    else:
        phase_ends = [1.0]
        phase_its = [iter(make_loader(gdata, list(w), True, ncfg, T, rows, data_seed,
                                      rank, world, nworkers, exclude_holdout=excl_holdout))]

    # gradient accumulation: global batch per GPU = rows * grad_accum. The swept setup is
    # 8 rows/GPU; running it as 4 x 2 HALVES activation memory (the 405M model at T=8192,
    # rows=8, depth=32 needs ~80GB of activations — over the GH200's 95GB once eval and
    # allocator overhead are counted) with identical training math.
    accum = cfg.get("grad_accum", 1)

    # auto-anneal, SYMMETRIC: the anneal starts when the held-out curve stalls — which can
    # be EARLIER than the planned step (don't burn budget on a flat curve) or LATER (don't
    # undertrain a model that is still learning; the stable phase extends past the planned
    # point up to hard_max_steps - decay_len). The decay window always keeps its planned
    # length. The decision persists in a marker file (survives chain-job restarts) and is
    # broadcast so all ranks switch at the same step. total_steps is thus a NOMINAL budget;
    # the actual end is anneal_start + decay_len, capped by hard_max_steps.
    planned_anneal = int(total * (1 - cfg.get("decay_frac", 0.2)))
    decay_len = total - planned_anneal
    hard_max = max(cfg.get("hard_max_steps", total), total)
    max_start = hard_max - decay_len          # latest possible anneal start
    auto = cfg.get("auto_anneal", True)
    min_start = int(total * cfg.get("min_anneal_frac", 0.35))
    marker = out / "anneal_start.json"
    anneal_start = None                        # None = not yet decided (stall or cap decides)
    if not auto:
        anneal_start = planned_anneal
    elif marker.exists():
        anneal_start = min(json.loads(marker.read_text())["step"], max_start)
        if rank == 0:
            print(f"auto-anneal marker: anneal from step {anneal_start}")

    model.train()
    t0 = time.time(); seen_chars = 0
    consec_skips = 0; total_skips = 0; diverged = False; stop_early = False
    end_step = (anneal_start if anneal_start is not None else max_start) + decay_len
    for step in range(step0, hard_max):
        if anneal_start is None and step >= max_start:
            anneal_start = max_start          # cap reached: deterministic on all ranks
        a0 = anneal_start if anneal_start is not None else max_start
        end_step = a0 + decay_len
        if step >= end_step:
            break                     # anneal finished: run is complete
        anneal = step >= a0
        # durable branch point: the last flat-LR state before the anneal. Lets us re-run a
        # LONGER or different anneal (data mix, decay shape, soup of several) afterwards
        # for 20% of the cost, instead of repeating the whole stable phase.
        if rank == 0 and step == a0 and not (out / "pre_anneal.pt").exists():
            core = model.module if is_ddp else model
            save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(), step=step, cfg=cfg),
                      out / "pre_anneal.pt")
            if not marker.exists():
                marker.write_text(json.dumps(dict(step=a0, reason="cap")))
            print(f"  saved pre_anneal.pt at step {step} (anneal branch point)", flush=True)
        if anneal:
            fr = (step - a0) / max(decay_len, 1)
            pi = 0
            while pi < len(phase_ends) - 1 and fr >= phase_ends[pi]:
                pi += 1
            if rank == 0 and pi != getattr(main, "_pi", -1):
                main._pi = pi
                print(f"  anneal phase {pi + 1}/{len(phase_ends)} from step {step} "
                      f"(mix {phases[pi][1] if phases else 'gold-only'})", flush=True)
            it = phase_its[pi]
        else:
            it = stable_it
        lr = cfg["lr"] * wsd_dyn(step, total, a0,
                                 cfg.get("warmup_frac", 0.04), cfg.get("decay_frac", 0.2))
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(it)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            # skip the allreduce on all but the last micro-batch. DDP reads the no_sync
            # flag during FORWARD, so the forward must run inside the context too.
            sync_ctx = (model.no_sync() if (is_ddp and micro < accum - 1)
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out_h = fwd(batch)
                    loss, logs = compute_loss(out_h, batch, lam=cfg.get("lam", 0.1))
                (loss / accum).backward()
            seen_chars += int((batch["seg_id"] > 0).sum()) * world   # real (non-pad) chars
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        # NaN-skip guard: one pathological batch must not poison the weights.
        # Check BEFORE opt.step() so non-finite grads are never applied.
        if torch.isfinite(gnorm):
            opt.step()
            consec_skips = 0
        else:
            consec_skips += 1; total_skips += 1
            if rank == 0:
                print(f"  [skip] non-finite grad at step {step} (consecutive={consec_skips}, "
                      f"total={total_skips})", flush=True)
            if consec_skips >= cfg.get("max_consec_skips", 25):
                if rank == 0:
                    print(f"  ABORT: {consec_skips} consecutive non-finite steps — real "
                          f"divergence, lower LR.", flush=True)
                diverged = True
                break

        if rank == 0 and (step % cfg.get("log_every", 20) == 0):
            dt = time.time() - t0
            mem = (round(torch.cuda.max_memory_allocated() / 2**30, 1)
                   if device.type == "cuda" else 0)
            rec = dict(step=step, lr=round(lr, 6), gnorm=round(float(gnorm), 3),
                       chps=round(seen_chars / dt / 1e6, 3), mem_gb=mem, anneal=anneal, **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as mf:
                mf.write(json.dumps(rec) + "\n")
        if step > step0 and step % cfg.get("ckpt_every", 2000) == 0:
            if rank == 0:
                core = model.module if is_ddp else model
                # step+1: this step is DONE — resume must continue at the next one
                save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(), step=step + 1, cfg=cfg), ckpt)
                if cfg.get("eval_every_ckpt", True):
                    try:
                        from eval.intrinsic import held_out_records, evaluate, restore_demo
                        if eval_shards:
                            from eval.val_eval import val_records
                            recs = val_records(eval_shards, cfg.get("eval_n", 256))
                        else:
                            recs = held_out_records(f"{gdata}/shards/v1_punct", cfg.get("eval_n", 256))
                        m = evaluate(core, recs, device)
                        m["demo"] = restore_demo(core, device)
                        m["step"] = step
                        m["split"] = eval_split
                        print("  EVAL " + json.dumps(m, ensure_ascii=False), flush=True)
                        with open(out / "eval.jsonl", "a") as ef:
                            ef.write(json.dumps(m, ensure_ascii=False) + "\n")
                        evs = [e for e in (json.loads(l) for l in open(out / "eval.jsonl")
                                           if "bits_per_char" in l)
                               if e.get("split", "train") == eval_split]
                        # best-checkpoint tracking: never lose the best held-out model,
                        # whatever the anneal tail does (stateless: derived from eval.jsonl)
                        if m["bits_per_char"] <= min(e["bits_per_char"] for e in evs):
                            import shutil
                            shutil.copyfile(ckpt, out / "best.pt")
                            print(f"  new best.pt (bpc {m['bits_per_char']})", flush=True)
                        # anneal early-stop: gold-only + decaying LR can tip into overfitting;
                        # if held-out hasn't improved for `anneal_patience` evals, stop —
                        # best.pt is the product.
                        # only judge the anneal from mid-decay on: WSD gains arrive when
                        # the LR falls through ~50%, and the gold-switch at near-peak LR
                        # causes a transient (observed +0.02 bpc at step 68k) that must
                        # not be mistaken for a stalled anneal.
                        pat = cfg.get("anneal_patience", 3)
                        if anneal and pat > 0 and step >= a0 + decay_len // 2:
                            ann = [e for e in evs if e["step"] > (anneal_start or 0)]
                            if len(ann) > pat:
                                bi = min(range(len(ann)), key=lambda i: ann[i]["bits_per_char"])
                                if len(ann) - 1 - bi >= pat:
                                    stop_early = True
                                    print(f"  ANNEAL EARLY-STOP: no held-out improvement in "
                                          f"{pat} evals (best at step {ann[bi]['step']}); "
                                          f"best.pt is the final model", flush=True)
                    except Exception as e:
                        print(f"  EVAL failed: {str(e)[:120]}", flush=True)
                # auto-anneal trigger: anneal the moment the held-out curve stalls
                if (auto and anneal_start is None and step >= min_start
                        and eval_stalled(out / "eval.jsonl",
                                         cfg.get("stall_window", 8), cfg.get("stall_eps", 0.002),
                                         split=eval_split)):
                    anneal_start = min(step + 1, max_start)
                    marker.write_text(json.dumps(dict(step=anneal_start, reason="stall",
                                                      triggered_at=step)))
                    print(f"  AUTO-ANNEAL: held-out bits/char stalled — annealing from step "
                          f"{anneal_start} (planned {planned_anneal}, cap {max_start}); "
                          f"run ends at {anneal_start + decay_len}", flush=True)
                elif auto and anneal_start is None and step == planned_anneal:
                    print(f"  EXTEND: still improving at planned anneal step {planned_anneal} "
                          f"— stable phase continues (anneal by step {max_start} at latest)",
                          flush=True)
            if is_ddp:
                import torch.distributed as dist
                t_a = torch.tensor([-1 if anneal_start is None else anneal_start,
                                    1 if stop_early else 0], dtype=torch.long, device=device)
                dist.broadcast(t_a, 0)
                v = int(t_a[0].item())
                anneal_start = None if v < 0 else v
                stop_early = bool(int(t_a[1].item()))
            if stop_early:
                break

    if rank == 0:
        if diverged:
            # do NOT write final.pt / advance last.pt: the run is not complete, and the
            # last good checkpoint is the thing to restart from (with a lower LR).
            print("ABORTED (divergence) — last good checkpoint left untouched")
        else:
            # step is stamped as end_step even on early stop so remaining chain jobs no-op;
            # stopped_at records the true last trained step. On early stop the model to USE
            # is best.pt (the early-stop criterion means the final weights are not the best).
            state = dict(model=(model.module if is_ddp else model).state_dict(),
                         opt=opt.state_dict(), step=end_step, cfg=cfg,
                         stopped_early=stop_early)
            save_ckpt(state, out / "final.pt")
            save_ckpt(state, ckpt)   # last.pt at the end step: remaining chain jobs no-op
            print(f"DONE (end step {end_step}, nominal budget {total}"
                  + (", EARLY-STOPPED — use best.pt" if stop_early else "") + ")")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    if diverged:
        sys.exit(1)


if __name__ == "__main__":
    main()
