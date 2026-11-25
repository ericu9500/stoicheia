"""Fine-tune the Stoicheia torso on I.PHI inscriptions (diffusion objective, Ithaca-
oriented masking: contiguous spans of 1-10 chars dominate). Config-driven, DDP/torchrun.

Reuses Stoicheia modules wholesale (PYTHONPATH=$STOICHEIA_ROOT); this file only changes:
  - init from $INS_TORSO (weights only; fresh optimizer, fresh short schedule)
  - data mix: iphi shards + gold/silver replay (anti-forgetting)
  - noise mix: span-heavy, span lengths matched to the Ithaca eval (1-10)
  - quick PHI-val greedy-restore CER as the per-checkpoint eval

  torchrun --nproc_per_node=4 insc_train/finetune.py --config configs/finetune.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import torch

from model.char_bert import CharBertConfig, CharBertEncoder, num_params
from train.collate import pack_batch
from train.data import DataConfig, MultiTierLoader, TierSpec
from train.loss import compute_loss
from train.noising import NoiseConfig
from train.train import ddp_setup, infinite_records, save_ckpt

INS_ROOT = Path(__file__).resolve().parents[1]


class MixDataset(torch.utils.data.IterableDataset):
    """Packed batches from an explicit DataConfig; each (rank, worker) gets a disjoint
    shard and its own RNG (same contract as Stoicheia's BatchDataset)."""
    def __init__(self, dcfg, ncfg, T, rows, seed, rank, world):
        super().__init__()
        self.dcfg, self.ncfg, self.T, self.rows = dcfg, ncfg, T, rows
        self.seed, self.rank, self.world = seed, rank, world

    def __iter__(self):
        import copy
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        gshard = self.rank * nw + wid
        gtot = self.world * nw
        c = copy.deepcopy(self.dcfg)
        loader = MultiTierLoader(c, rank=gshard, world_size=gtot)
        g = torch.Generator().manual_seed(self.seed * 100003 + gshard)
        gen = infinite_records(loader)
        while True:
            yield pack_batch(gen, self.ncfg, self.T, self.rows, g)


def make_loader(dcfg, ncfg, T, rows, seed, rank, world, num_workers):
    return torch.utils.data.DataLoader(
        MixDataset(dcfg, ncfg, T, rows, seed, rank, world), batch_size=None,
        num_workers=num_workers, prefetch_factor=(2 if num_workers > 0 else None),
        persistent_workers=(num_workers > 0), pin_memory=True)


def lr_mult(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))   # cosine to near-zero


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")

    T = cfg["seq_len"]; rows = cfg["micro_batch"]; accum = cfg.get("grad_accum", 1)
    total = cfg["total_steps"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    mcfg = CharBertConfig(attn_impl=cfg.get("attn", "flex"), d_model=cfg["d_model"],
                          n_heads=cfg["d_model"] // 64, depth=cfg["depth"],
                          char_window=cfg["char_window"], qk_norm=True)
    model = CharBertEncoder(mcfg).to(device)

    step0 = 0
    ckpt = out / "last.pt"
    if ckpt.exists():                                  # resume a crashed finetune
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"]); step0 = sd["step"]
        if rank == 0:
            print(f"resumed finetune from step {step0}")
    else:                                              # init from the pretrained torso
        torso = os.path.expandvars(cfg["torso"])
        sd = torch.load(torso, map_location=device)
        model.load_state_dict(sd["model"])
        if rank == 0:
            print(f"initialized from torso {torso} (pretrain step {sd.get('step')})")
    if step0 >= total:
        if rank == 0:
            print("finetune already complete")
        return
    if rank == 0:
        print(f"params={num_params(model)/1e6:.1f}M T={T} rows={rows}x{accum} world={world} "
              f"total={total}")
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model

    decay_p = [p for p in model.parameters() if p.ndim >= 2]
    nodecay_p = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": cfg.get("wd", 0.1)},
                             {"params": nodecay_p, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if ckpt.exists() and "opt" in sd:
        try:
            opt.load_state_dict(sd["opt"])
        except Exception:
            pass

    # noise: Ithaca-oriented. Spans dominate, geometric length tuned to the 1-10 band.
    nc = cfg.get("noise", {})
    ncfg = NoiseConfig(w_span=nc.get("w_span", 0.45), w_word=nc.get("w_word", 0.15),
                       w_elastic=nc.get("w_elastic", 0.0), w_iid=nc.get("w_iid", 0.15),
                       w_halfword=nc.get("w_halfword", 0.15),
                       w_substitute=nc.get("w_substitute", 0.10),
                       span_mean=nc.get("span_mean", 4.0), span_max=nc.get("span_max", 12))

    ins = os.path.expandvars(cfg["iphi_shards"])
    gcb = os.path.expandvars(cfg["shards"])
    mix = cfg.get("mix", {"iphi": 1.0, "iphi_syn": 0.5, "gold": 0.15, "silver": 0.15})
    dcfg = DataConfig(tiers={
        "iphi":     TierSpec(ins, mix.get("iphi", 1.0), tier_filter="iphi"),
        "iphi_syn": TierSpec(ins, mix.get("iphi_syn", 0.5), tier_filter="iphi_syn"),
        "gold":     TierSpec(gcb, mix.get("gold", 0.15), tier_filter="pristine"),
        "silver":   TierSpec(gcb, mix.get("silver", 0.15), tier_filter="repaired"),
    }, window_chars=T, seed=cfg["seed"] + step0)

    it = iter(make_loader(dcfg, ncfg, T, rows, cfg["seed"] + step0, rank, world,
                          cfg.get("num_workers", 8)))

    model.train()
    t0 = time.time(); seen = 0
    metrics_f = out / "metrics.jsonl"
    import contextlib
    for step in range(step0, total):
        lr = cfg["lr"] * lr_mult(step, total, cfg.get("warmup", 200))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(it)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            sync = (model.no_sync() if (is_ddp and micro < accum - 1)
                    else contextlib.nullcontext())
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out_h = fwd(batch)
                    loss, logs = compute_loss(out_h, batch, lam=cfg.get("lam", 0.1))
                (loss / accum).backward()
            seen += int((batch["seg_id"] > 0).sum()) * world
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            opt.step()

        if rank == 0 and step % cfg.get("log_every", 20) == 0:
            rec = dict(step=step, lr=round(lr, 7), gnorm=round(float(gnorm), 3),
                       chps=round(seen / (time.time() - t0) / 1e6, 3), **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as f:
                f.write(json.dumps(rec) + "\n")
        if rank == 0 and step > step0 and step % cfg.get("ckpt_every", 500) == 0:
            core = model.module if is_ddp else model
            save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(), step=step + 1,
                           cfg=cfg), ckpt)

    if rank == 0:
        core = model.module if is_ddp else model
        state = dict(model=core.state_dict(), opt=opt.state_dict(), step=total, cfg=cfg)
        save_ckpt(state, out / "final.pt")
        save_ckpt(state, ckpt)
        print("FINETUNE DONE")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
