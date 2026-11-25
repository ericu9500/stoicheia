"""Fine-tune Stoicheia into a macronizer + metrical scanner.

Single-process or torchrun/DDP (data-parallel over packed rows; every rank holds the
data and takes a disjoint slice of rows each epoch).

  torchrun --nproc_per_node=4 -m meter.train --config configs/joint_pilot.json

Epoch composition (identical on every rank, epoch-seeded):
  macron verse sources: every record, every epoch
  macron prose (OGA silver): a fresh random sample of `prose_per_epoch` records
  scansion: fresh multi-verse windows over the train works (`scan_passes` passes)
Dev = dev_aristophanes (macron balanced acc) + whole verses of the scanner dev works.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from meter.backbone import load_backbone
from meter.dataset import batch_rows, load_records, make_windows, pack_records
from meter.model import MeterConfig, MeterModel

SCAN_DEV_WORKS = {"odyssey9", "isthmians", "seven"}
SCAN_TEST_WORKS = {"iliad22", "pythians", "persians", "theognis"}

# sweep1_chol is omitted: it is byte-identical to babrius_chol
VERSE_SOURCES = ["hypotactic", "drama_ia6", "drama_ia6_tet", "anthology",
                 "nonnus_quintus", "babrius_chol", "theocritus_doric",
                 "theocritus_other", "sweep1_hex", "sweep1_ia6", "sweep1_eleg"]
PROSE_SOURCES = ["oga_0", "oga_1", "oga_2", "oga_3"]


def ddp_setup():
    if "RANK" in os.environ:
        import datetime
        import torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, True
    return 0, 1, False


def param_groups(model: MeterModel, cfg):
    """Flat head LR, flat encoder LR (LLRD was flat/harmful in the tagger project).
    Weight decay on matrices only (pretraining convention)."""
    lr_enc, lr_head, wd = cfg["lr_enc"], cfg["lr_head"], cfg.get("wd", 0.01)
    groups = {}

    def add(p, lr, is_enc=True):
        key = (lr, 0.0 if p.ndim < 2 else wd, is_enc)
        groups.setdefault(key, []).append(p)

    enc = model.encoder
    for m in (enc.e_char, enc.e_bnd, enc.e_dia, enc.e_punct, *enc.blocks,
              enc.norm_out, enc.head_char, enc.head_bnd, enc.head_dia,
              enc.head_cap, enc.head_punct):
        for p in m.parameters():
            add(p, lr_enc)
    for m in (model.head_mac, model.head_scan):
        for p in m.parameters():
            add(p, lr_head, is_enc=False)
    if hasattr(model, "mix_w"):
        add(model.mix_w, lr_head, is_enc=False)
    cap_emb = getattr(enc, "cap_emb", None)
    if cap_emb is not None:
        for p in cap_emb.parameters():
            add(p, lr_head)
    return [dict(params=ps, lr=lr, weight_decay=w, base_lr=lr, is_enc=e)
            for (lr, w, e), ps in groups.items()]


@torch.no_grad()
def evaluate_dev(model, dev_rows, dev_records, micro, device, T):
    """Counts tensor for cross-rank reduction:
    [0:4]  mac long_ok, long_n, short_ok, short_n
    [4:12] scan per-class ok,n for classes 0..3
    [12:15] end-detection tp, fp, fn  (end = class in {1,2,3})
    """
    model.eval()
    cnt = torch.zeros(15, dtype=torch.long, device=device)
    for i in range(0, len(dev_rows), micro):
        batch = batch_rows(dev_rows[i:i + micro], dev_records, T)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        pm = out["mac"].argmax(-1)
        for cls, off in ((0, 0), (1, 2)):
            m = batch["y_mac"] == cls
            cnt[off] += int(((pm == cls) & m).sum())
            cnt[off + 1] += int(m.sum())
        ps = out["scan"].argmax(-1)
        ys = batch["y_scan"]
        for cls in range(4):
            m = ys == cls
            cnt[4 + 2 * cls] += int(((ps == cls) & m).sum())
            cnt[4 + 2 * cls + 1] += int(m.sum())
        valid = ys != -100
        g_end = (ys > 0) & valid
        p_end = (ps > 0) & valid
        cnt[12] += int((g_end & p_end).sum())
        cnt[13] += int((~g_end & p_end).sum())
        cnt[14] += int((g_end & ~p_end).sum())
    model.train()
    return cnt


def dev_metrics(cnt):
    c = cnt.tolist()
    m = {}
    if c[1] and c[3]:
        m["mac_bal"] = round((c[0] / c[1] + c[2] / c[3]) / 2, 4)
        m["mac_acc"] = round((c[0] + c[2]) / (c[1] + c[3]), 4)
        m["mac_n"] = c[1] + c[3]
    recalls = [c[4 + 2 * k] / c[4 + 2 * k + 1] for k in range(4) if c[4 + 2 * k + 1]]
    tot = sum(c[4 + 2 * k + 1] for k in range(4))
    if tot:
        m["scan_bal"] = round(sum(recalls) / len(recalls), 4)
        m["scan_acc"] = round(sum(c[4 + 2 * k] for k in range(4)) / tot, 4)
        tp, fp, fn = c[12], c[13], c[14]
        m["end_f1"] = round(2 * tp / max(2 * tp + fp + fn, 1), 4)
        m["scan_n"] = tot
    return m


def selection_score(m, select):
    if select == "mac":
        return m.get("mac_bal", -1.0)
    if select == "scan":
        return (m.get("scan_bal", 0) + m.get("end_f1", 0)) / 2 if "scan_bal" in m else -1.0
    return (m.get("mac_bal", 0) + m.get("scan_bal", 0) + m.get("end_f1", 0)) / 3


def auto_class_weights(records, attr, k):
    cnt = np.zeros(k, dtype=np.int64)
    for r in records:
        y = getattr(r, attr)
        for c in range(k):
            cnt[c] += int((y == c).sum())
    total = cnt.sum()
    return [float(total / (k * c)) if c else 1.0 for c in cnt], cnt.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.ckpt:
        cfg["ckpt"] = a.ckpt
    for k in ("out_dir", "ckpt", "encoded"):
        cfg[k] = os.path.expandvars(cfg[k])

    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0) + rank)
    torch.set_float32_matmul_precision("high")

    T, micro = cfg.get("T", 2048), cfg["micro_batch"]
    w_mac, w_scan = cfg.get("w_mac", 1.0), cfg.get("w_scan", 1.0)
    enc_dir = Path(cfg["encoded"])
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    # ---- data (identical on every rank)
    t0 = time.time()
    verse, prose, scan_train_verses, dev_records = [], [], [], []
    if w_mac > 0:
        for name in cfg.get("verse_sources", VERSE_SOURCES):
            p = enc_dir / f"{name}.npz"
            if p.exists():
                verse += load_records(p)[0]
        for name in cfg.get("prose_sources", PROSE_SOURCES):
            p = enc_dir / f"{name}.npz"
            if p.exists():
                prose += load_records(p)[0]
        dev_records += load_records(enc_dir / "dev_aristophanes.npz")[0]
    scan_dev_records = []
    if w_scan > 0:
        recs, works = load_records(enc_dir / "scan_corpus.npz")
        by_work = {}
        for r, w in zip(recs, works):
            by_work.setdefault(w, []).append(r)
        for w, rs in sorted(by_work.items()):
            if w in SCAN_TEST_WORKS:
                continue
            elif w in SCAN_DEV_WORKS:
                scan_dev_records += rs
            else:
                scan_train_verses.append(rs)
    dev_records += scan_dev_records
    if cfg.get("max_records"):   # smoke tests
        n = cfg["max_records"]
        verse = verse[:n]
        prose = prose[:n]
        scan_train_verses = [rs[:n // 4] for rs in scan_train_verses[:4]]
        dev_records = dev_records[:n]

    mac_w = cfg.get("mac_class_w", "auto")
    scan_w = cfg.get("scan_class_w", "auto")
    if mac_w == "auto":
        mac_w, mac_cnt = auto_class_weights(verse + prose[:200000], "y_mac", 2) \
            if (verse or prose) else ([1.0, 1.0], [0, 0])
    if scan_w == "auto":
        flat = [r for rs in scan_train_verses for r in rs[:2000]]
        scan_w, scan_cnt = auto_class_weights(flat, "y_scan", 4) if flat \
            else ([1.0] * 4, [0] * 4)
    if rank == 0:
        print(f"data: verse={len(verse):,} prose={len(prose):,} "
              f"scan_works={len(scan_train_verses)} dev={len(dev_records):,} "
              f"mac_w={[round(x,3) for x in mac_w]} scan_w={[round(x,3) for x in scan_w]} "
              f"load={time.time()-t0:.0f}s", flush=True)

    dev_rows, _ = pack_records(dev_records, T)
    dev_shard = dev_rows[rank::world]

    # ---- model
    encoder, pcfg = load_backbone(cfg["ckpt"], device, attn_impl=cfg.get("attn", "sdpa"))
    mcfg = MeterConfig(head_dropout=cfg.get("head_dropout", 0.33),
                       w_mac=w_mac, w_scan=w_scan,
                       use_cap=cfg.get("use_cap", True),
                       scalar_mix=cfg.get("scalar_mix", True),
                       mac_class_w=mac_w, scan_class_w=scan_w)
    model = MeterModel(encoder, mcfg).to(device)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    core = model.module if is_ddp else model

    opt = torch.optim.AdamW(
        [g for g in param_groups(core, cfg) if any(p.requires_grad for p in g["params"])],
        betas=(0.9, 0.95), fused=(device.type == "cuda"))

    # ---- steps/epoch from a reference epoch-0 assembly
    # phase 2 ("curriculum"): after cfg[epochs] mixed epochs, p2_epochs more with a
    # different prose budget (default 0 = verse-only), LR continuing the same cosine
    p2_epochs = cfg.get("p2_epochs", 0)

    def epoch_records(epoch):
        rng = np.random.default_rng(cfg.get("seed", 0) * 1000 + epoch)
        recs = list(verse)
        base_prose = cfg.get("prose_per_epoch", 150000)
        if epoch >= cfg["epochs"]:
            base_prose = cfg.get("p2_prose_per_epoch", 0)
        n_p = min(base_prose, len(prose))
        if n_p and w_mac > 0:
            recs += [prose[i] for i in rng.choice(len(prose), n_p, replace=False)]
        if w_scan > 0:
            for rs in scan_train_verses:
                recs += make_windows(rs, rng, cfg.get("scan_passes", 2),
                                     cfg.get("scan_max_verses", 8), T)
        order = rng.permutation(len(recs))
        rows, _ = pack_records(recs, T, order)
        return recs, rows

    ref_recs, ref_rows = epoch_records(0)
    accum = int(cfg.get("grad_accum", 1))      # gradient accumulation (single-GPU runs keep the global batch)
    steps_per_epoch = max(len(ref_rows) // world // (micro * accum), 1)
    total_steps = steps_per_epoch * cfg["epochs"]
    if p2_epochs:
        _, p2_rows = epoch_records(cfg["epochs"])
        total_steps += max(len(p2_rows) // world // (micro * accum), 1) * p2_epochs
    warmup = max(int(total_steps * cfg.get("warmup_frac", 0.1)), 1)
    if rank == 0:
        print(f"rows/epoch~{len(ref_rows)} steps/epoch~{steps_per_epoch} "
              f"total~{total_steps} p2_epochs={p2_epochs}", flush=True)

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        f = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    metrics_f = out / f"metrics_rank{rank}.jsonl"
    t0 = time.time()
    best_score, best_epoch = -1.0, -1
    step = 0
    epoch0 = 0
    last = out / "last.pt"          # epoch-level resume
    if last.exists():
        sd = torch.load(last, map_location=device, weights_only=False)
        core.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        epoch0, step = sd["epoch"] + 1, sd["step"]
        best_score, best_epoch = sd["best_score"], sd["best_epoch"]
        if rank == 0:
            print(f"resumed from last.pt: epoch {epoch0}, step {step}, best {best_score:.4f}@{best_epoch}", flush=True)
    for epoch in range(epoch0, cfg["epochs"] + p2_epochs):
        recs, rows = (ref_recs, ref_rows) if epoch == 0 else epoch_records(epoch)
        n = len(rows) // world * world
        shard = rows[rank:n:world]
        chunk = micro * accum
        nsteps = len(shard) // chunk * chunk
        for i in range(0, nsteps, chunk):
            lr_s = lr_scale(step)
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * lr_s
            opt.zero_grad(set_to_none=True)
            for j in range(0, chunk, micro):
                batch = batch_rows(shard[i + j:i + j + micro], recs, T, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    out_h = model(batch)
                    loss, logs = core.loss(out_h, batch)
                (loss / accum).backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
            if torch.isfinite(gnorm):
                opt.step()
            elif rank == 0:
                print(f"  [skip] non-finite grad at step {step}", flush=True)
            if rank == 0 and step % cfg.get("log_every", 20) == 0:
                rec = dict(epoch=epoch, step=step,
                           lr=round(opt.param_groups[0]["lr"], 7),
                           loss=round(loss.item(), 4), gnorm=round(float(gnorm), 3),
                           sps=round(step / max(time.time() - t0, 1e-9), 2), **logs)
                print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
                with open(metrics_f, "a") as mf:
                    mf.write(json.dumps(rec) + "\n")
            step += 1

        cnt = evaluate_dev(model, dev_shard, dev_records, cfg.get("eval_micro", micro),
                           device, T)
        if is_ddp:
            import torch.distributed as dist
            dist.all_reduce(cnt)
        m = dev_metrics(cnt)
        score = selection_score(m, cfg.get("select", "mac"))
        if rank == 0:
            print(f"  EVAL {json.dumps(dict(epoch=epoch, **m))}", flush=True)
            with open(out / "eval.jsonl", "a") as ef:
                ef.write(json.dumps(dict(epoch=epoch, **m)) + "\n")
            if score > best_score:
                tmp = out / "best.pt.tmp"
                torch.save(dict(model=core.state_dict(), mcfg=vars(mcfg), cfg=cfg,
                                pretrain_cfg=pcfg, epoch=epoch, dev=m, T=T), tmp)
                os.replace(tmp, out / "best.pt")
                print(f"  new best.pt (score {score:.4f})", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
        if rank == 0:
            tmp = out / "last.pt.tmp"
            torch.save(dict(model=core.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step,
                            best_score=best_score, best_epoch=best_epoch), tmp)
            os.replace(tmp, last)
        if epoch - best_epoch >= cfg.get("patience", 6):
            if rank == 0:
                print(f"EARLY STOP at epoch {epoch} (best epoch {best_epoch}, "
                      f"score {best_score:.4f})", flush=True)
            break

    if rank == 0:
        print(f"DONE best_score={best_score:.4f} best_epoch={best_epoch} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
