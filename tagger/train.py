"""Fine-tune Stoicheia into a joint lemmatizer + XPOS tagger.

Single-process or torchrun/DDP (data-parallel over packed rows; the dataset is tiny,
every rank holds it all in RAM and takes a disjoint slice of rows each epoch).

  torchrun --nproc_per_node=4 -m tagger.train --config configs/tagger_fold0.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tagger.backbone import load_backbone_auto
from tagger.conllu import Sentence, Token, read_conllu
from tagger.dataset import TaggerDataset, batch_chunk, encode_word, pack_dev_items
from tagger.edits import LabelVocab
from tagger.model import TaggerConfig, TaggerModel


def read_silver(path, limit_words):
    """Qwen-teacher silver lemmas (one word/line jsonl grouped by 'sentence') ->
    Sentence objects with dummy XPOS/UPOS; used for lemma-distillation pretraining."""
    sents, cur_s, cur = [], None, []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            s, fo, le = r.get("sentence", ""), r.get("form", ""), r.get("lemma", "")
            if not fo or not le:
                continue
            if s != cur_s:
                if cur:
                    sents.append(Sentence(tokens=cur))
                cur_s, cur = s, []
            cur.append(Token(str(len(cur) + 1), fo, le, "-", "-" * 9,
                             "_", "_", "_", "_", "_"))
            n += 1
            if n >= limit_words:
                break
    if cur:
        sents.append(Sentence(tokens=cur))
    return sents


def mask_non_lemma_labels(ds: TaggerDataset):
    """Silver has trustworthy lemmas only: keep y_script, ignore xpos/upos/full-tag."""
    for e in ds.encs:
        if e is None:
            continue
        e.y_xpos[:] = -100
        e.y_upos[:] = -100
        e.y_tag[:] = -100


def ddp_setup():
    if "RANK" in os.environ:
        import datetime
        import torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, True
    return 0, 1, False


def param_groups(model: TaggerModel, cfg):
    """Head LR flat; encoder LR with layerwise decay from the top block down.
    Decay only matrices (same convention as pretraining).

    Dispatches on the encoder type: an HF backbone (tagger.hf_backbone.HFBackboneWithHidden)
    has none of CharBertWithHidden's e_char/e_bnd/e_dia/e_punct embeddings or
    head_char/head_bnd/... pretraining heads, so it gets its own (simpler) walk in
    tagger.hf_backbone.param_groups_hf; this keeps one call site for both tagger/train.py and
    parser/joint_train.py (via `from tagger.train import param_groups as tagger_param_groups`)
    regardless of which backbone a config selects."""
    from tagger.hf_backbone import HFBackboneWithHidden, param_groups_hf
    if isinstance(model.encoder, HFBackboneWithHidden):
        return param_groups_hf(model, cfg)

    llrd = cfg.get("llrd", 0.95)
    lr_enc, lr_head, wd = cfg["lr_enc"], cfg["lr_head"], cfg.get("wd", 0.01)
    depth = len(model.encoder.blocks)
    groups = {}

    def add(p, lr, is_enc=True):
        key = (lr, 0.0 if p.ndim < 2 else wd, is_enc)
        groups.setdefault(key, []).append(p)

    enc = model.encoder
    emb_lr = lr_enc * llrd ** depth
    for m in (enc.e_char, enc.e_bnd, enc.e_dia, enc.e_punct):
        for p in m.parameters():
            add(p, emb_lr)
    for i, blk in enumerate(enc.blocks):
        for p in blk.parameters():
            add(p, lr_enc * llrd ** (depth - 1 - i))
    for p in enc.norm_out.parameters():
        add(p, lr_enc)
    # pretraining output heads ride along untrained-into-loss; give them enc LR (harmless)
    for m in (enc.head_char, enc.head_bnd, enc.head_dia, enc.head_cap, enc.head_punct):
        for p in m.parameters():
            add(p, lr_enc)
    heads = [model.xpos_heads, model.head_script, model.head_upos]
    if model.head_flat is not None:
        heads.append(model.head_flat)
    for m in heads:
        for p in m.parameters():
            add(p, lr_head, is_enc=False)
    if hasattr(model, "mix_w"):
        add(model.mix_w, lr_head, is_enc=False)
    cap_emb = getattr(enc, "cap_emb", None)
    if cap_emb is not None:                    # fine-tune-only channel: learns at head LR
        for p in cap_emb.parameters():
            add(p, lr_head)
    return [dict(params=ps, lr=lr, weight_decay=w, base_lr=lr, is_enc=e)
            for (lr, w, e), ps in groups.items()]


@torch.no_grad()
def evaluate_dev(model, dev_rows, micro, device, T, W, tokenizer=None):
    """Unconstrained dev accuracies (early-stop signal; full constrained decode is
    evaluate.py's job). Returns counts tensor for cross-rank reduction."""
    model.eval()
    # counts: [xpos_exact_ok, script_ok, upos_ok, n_words, pos1_ok]
    cnt = torch.zeros(5, dtype=torch.long, device=device)
    for i in range(0, len(dev_rows), micro):
        batch = batch_chunk(dev_rows[i:i + micro], T, W, tokenizer)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        m = batch["y_script"] != -100
        if not m.any():
            continue
        if "flat" in out:
            cnt[0] += int(((out["flat"].argmax(-1) == batch["y_tag"]) & m).sum())
            cnt[4] += int(((out["xpos"][0].argmax(-1) == batch["y_xpos"][:, :, 0]) & m).sum())
        else:
            ok = torch.ones_like(m)
            for p, lg in enumerate(out["xpos"]):
                pred = lg.argmax(-1)
                good = pred == batch["y_xpos"][:, :, p]
                ok &= good | (batch["y_xpos"][:, :, p] == -100)
                if p == 0:
                    cnt[4] += int((good & m).sum())
            cnt[0] += int((ok & m).sum())
        cnt[1] += int(((out["script"].argmax(-1) == batch["y_script"]) & m).sum())
        cnt[2] += int(((out["upos"].argmax(-1) == batch["y_upos"]) & m).sum())
        cnt[3] += int(m.sum())
    model.train()
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="override backbone checkpoint path")
    ap.add_argument("--fold", type=int, default=None)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.ckpt:
        cfg["ckpt"] = a.ckpt
    if a.fold is not None:
        cfg["fold"] = a.fold
    for k in ("out_dir", "kfold_dir"):
        cfg[k] = os.path.expandvars(cfg[k])
    if "ckpt" in cfg:
        cfg["ckpt"] = os.path.expandvars(cfg["ckpt"])

    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0) + rank)
    torch.set_float32_matmul_precision("high")

    fold = cfg["fold"]
    T, W, micro = cfg.get("T", 2048), cfg.get("W", 384), cfg["micro_batch"]
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    # ---- data + vocab (every rank builds identically; deterministic)
    t0 = time.time()
    kdir = Path(cfg["kfold_dir"])
    train_sents = list(read_conllu(kdir / f"train{fold}.conllu"))
    dev_sents = list(read_conllu(kdir / f"dev{fold}.conllu"))
    if cfg.get("max_sents"):
        train_sents = train_sents[:cfg["max_sents"]]
        dev_sents = dev_sents[:max(cfg["max_sents"] // 4, 8)]
    vocab = LabelVocab.build(train_sents, lambda f: encode_word(f) is not None)
    if rank == 0:
        vocab.save(out / "vocab.json")
        print(f"fold={fold} train_sents={len(train_sents)} dev_sents={len(dev_sents)} "
              f"scripts={vocab.n_scripts} tags={len(vocab.tags)} "
              f"vocab_build={time.time()-t0:.0f}s", flush=True)

    # ---- model (loaded before the datasets since the HF path needs its tokenizer to encode)
    encoder, pcfg, tokenizer = load_backbone_auto(cfg, device)
    is_hf = tokenizer is not None
    hf_max_len = cfg.get("hf_max_len", 512)
    use_cap = cfg.get("use_cap", False)
    if is_hf and use_cap:
        # the fine-tune-only capitalization channel is injected into CharBERT's char-plane
        # embeddings (see CharBertWithHidden.forward); there's no equivalent injection point
        # for subword input (casing is already whatever the subword vocab encodes), so it's
        # forced off here rather than silently leaving a dead, ungraded nn.Embedding around
        print("  [hf backbone] ignoring use_cap=true (no char-plane to inject into)",
              flush=True)
        use_cap = False

    train_ds = TaggerDataset(train_sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=hf_max_len)
    dev_ds = TaggerDataset(dev_sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=hf_max_len)
    dev_rows, _ = pack_dev_items(dev_ds.encs, W, tokenizer, T)
    dev_shard = dev_rows[rank::world]
    if rank == 0:
        print(f"encoded in {time.time()-t0:.0f}s  dev_rows={len(dev_rows)}", flush=True)

    tcfg = TaggerConfig(pool=cfg.get("pool", "mean"),
                        head_dropout=cfg.get("head_dropout", 0.1),
                        w_xpos=cfg.get("w_xpos", 1.0), w_script=cfg.get("w_script", 1.0),
                        w_upos=cfg.get("w_upos", 0.2), use_cap=use_cap,
                        w_flat=cfg.get("w_flat", 0.0),
                        scalar_mix=cfg.get("scalar_mix", False))
    model = TaggerModel(encoder, vocab, tcfg, W=W).to(device)
    if cfg.get("freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    core = model.module if is_ddp else model

    opt = torch.optim.AdamW(
        [g for g in param_groups(core, cfg) if any(p.requires_grad for p in g["params"])],
        betas=(0.9, 0.95), fused=(device.type == "cuda"))

    # steps/epoch from a reference packing; schedule = warmup + cosine over the plan
    ref_rows, trunc = pack_dev_items(train_ds.encs, W, tokenizer, T)
    n_rows = len(ref_rows) // world * world
    steps_per_epoch = max(n_rows // world // micro, 1)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(int(total_steps * cfg.get("warmup_frac", 0.05)), 1)
    if rank == 0:
        print(f"rows/epoch={n_rows} steps/epoch={steps_per_epoch} total={total_steps} "
              f"truncated_sents={trunc}", flush=True)

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        f = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    metrics_f = out / f"metrics_rank{rank}.jsonl"
    t0 = time.time()

    def do_step(batch, lr_s, step, tag, enc_scale=1.0):
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * lr_s * (enc_scale if pg.get("is_enc") else 1.0)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out_h = model(batch)
            loss, logs = core.loss(out_h, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        if torch.isfinite(gnorm):
            opt.step()
        elif rank == 0:
            print(f"  [skip] non-finite grad at {tag} step {step}", flush=True)
        if rank == 0 and step % cfg.get("log_every", 20) == 0:
            rec = dict(phase=tag, step=step, lr=round(opt.param_groups[0]["lr"], 7),
                       loss=round(loss.item(), 4), gnorm=round(float(gnorm), 3),
                       sps=round(step / max(time.time() - t0, 1e-9), 2), **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as mf:
                mf.write(json.dumps(rec) + "\n")

    # ---- phase 1 (optional): silver lemma-distillation pretrain (script loss only)
    if cfg.get("silver"):
        t1 = time.time()
        silver_sents = read_silver(os.path.expandvars(cfg["silver"]),
                                   cfg.get("silver_limit", 4_000_000))
        silver_ds = TaggerDataset(silver_sents, vocab, T, W, tokenizer=tokenizer,
                                  hf_max_len=hf_max_len)
        mask_non_lemma_labels(silver_ds)
        if rank == 0:
            print(f"silver: {len(silver_sents)} sents, encoded in {time.time()-t1:.0f}s",
                  flush=True)
        sstep = 0
        for sep in range(cfg.get("silver_epochs", 1)):
            rows, _ = pack_dev_items(silver_ds.encs, W, tokenizer, T,
                                     order=torch.randperm(
                                         len(silver_ds.encs),
                                         generator=torch.Generator().manual_seed(123 + sep)
                                         ).tolist())
            n = len(rows) // world * world
            shard = rows[rank:n:world]
            nsteps = len(shard) // micro * micro
            warm = max((nsteps // micro) // 10, 1)
            for i in range(0, nsteps, micro):
                # warmup then constant LR (distillation phase, no decay); optionally keep
                # the encoder frozen so silver noise cannot drift the tagging features
                do_step(batch_chunk(shard[i:i + micro], T, W, tokenizer),
                        min((sstep + 1) / warm, 1.0), sstep, "silver",
                        enc_scale=cfg.get("silver_enc_scale", 0.0))
                sstep += 1
        cnt = evaluate_dev(model, dev_shard, cfg.get("eval_micro", micro), device, T, W,
                           tokenizer)
        if is_ddp:
            import torch.distributed as dist
            dist.all_reduce(cnt)
        if rank == 0:
            n_w = max(int(cnt[3]), 1)
            print(f"  SILVER DONE steps={sstep} dev_script_acc={int(cnt[1])/n_w:.4f}",
                  flush=True)

    best_score, best_epoch = -1.0, -1
    step = 0
    for epoch in range(cfg["epochs"]):
        rows, _ = pack_dev_items(train_ds.encs, W, tokenizer, T,
                                 order=torch.randperm(
                                     len(train_ds.encs),
                                     generator=torch.Generator().manual_seed(
                                         cfg.get("seed", 0) * 1000 + epoch)).tolist())
        n = len(rows) // world * world
        shard = rows[rank:n:world]
        nsteps = len(shard) // micro * micro
        for i in range(0, nsteps, micro):
            do_step(batch_chunk(shard[i:i + micro], T, W, tokenizer), lr_scale(step), step,
                    "gold")
            step += 1

        # ---- dev eval (each rank its shard, reduce counts)
        cnt = evaluate_dev(model, dev_shard, cfg.get("eval_micro", micro), device, T, W,
                           tokenizer)
        if is_ddp:
            import torch.distributed as dist
            dist.all_reduce(cnt)
        n_w = max(int(cnt[3]), 1)
        m = dict(epoch=epoch, dev_xpos_exact=round(int(cnt[0]) / n_w, 4),
                 dev_script_acc=round(int(cnt[1]) / n_w, 4),
                 dev_upos_acc=round(int(cnt[2]) / n_w, 4),
                 dev_pos1_acc=round(int(cnt[4]) / n_w, 4), n_words=n_w)
        score = (m["dev_xpos_exact"] + m["dev_script_acc"]) / 2
        if rank == 0:
            print("  EVAL " + json.dumps(m), flush=True)
            with open(out / "eval.jsonl", "a") as ef:
                ef.write(json.dumps(m) + "\n")
            if score > best_score:
                tmp = out / "best.pt.tmp"
                torch.save(dict(model=core.state_dict(), tcfg=vars(tcfg), cfg=cfg,
                                pretrain_cfg=pcfg, epoch=epoch, dev=m, W=W, T=T), tmp)
                os.replace(tmp, out / "best.pt")
                print(f"  new best.pt (score {score:.4f})", flush=True)
        # keep best/stop decisions consistent across ranks (same reduced counts)
        if score > best_score:
            best_score, best_epoch = score, epoch
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
