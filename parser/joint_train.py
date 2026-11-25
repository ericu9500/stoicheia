"""Fine-tune ONE Stoicheia backbone into a full CoNLL-U predictor: lemma (edit-script),
UPOS, factored XPOS/morph, and biaffine HEAD+DEPREL — all heads share the pooled, scalar-mixed
word representation and train jointly (multi-task; morph and syntax reinforce each other).

Reuses the tagger's proven data pipeline (TaggerDataset/LabelVocab, SOTA 94.22 XPOS) and DDP
recipe, adds a Dozat-Manning biaffine head + per-sentence gold alignment + LAS/UAS eval.

  torchrun --nproc_per_node=4 -m parser.joint_train --config configs/joint.json
"""
from __future__ import annotations

import argparse, datetime, json, math, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tagger.backbone import load_backbone_auto
from tagger.conllu import read_conllu
from tagger.dataset import TaggerDataset, batch_chunk, encode_word, pack_dev_items
from tagger.edits import LabelVocab
from tagger.model import TaggerConfig
from tagger.train import param_groups as tagger_param_groups

from parser.biaffine import ParserConfig
from parser.labels import DeprelVocab
from parser.model import build_gold
from parser.joint_model import JointModel


def ddp_setup():
    if "RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank(); world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world, True
    return 0, 1, False


def parse_gold(sent_ids, sentences, deprel_vocab, max_w, device):
    """Gold heads/labels/mask (n_sent, max_w) for the sentences in `sent_ids`, in the same
    encodable-token order JointModel._regroup used (build_gold shares that ordering)."""
    n_sent = len(sent_ids)
    heads = torch.full((n_sent, max_w), -100, dtype=torch.long)
    labels = torch.full((n_sent, max_w), -100, dtype=torch.long)
    mask = torch.zeros((n_sent, max_w), dtype=torch.bool)
    for local, si in enumerate(sent_ids):
        n, h, l = build_gold(sentences[si], deprel_vocab)
        n = min(n, max_w)
        if n == 0:
            continue
        heads[local, :n] = torch.tensor(h[:n])
        labels[local, :n] = torch.tensor(l[:n])
        mask[local, :n] = True
    return heads.to(device), labels.to(device), mask.to(device)


@torch.no_grad()
def evaluate_dev(model, core, dev_rows, sentences, deprel_vocab, micro, device, T, W,
                 decode="greedy", tokenizer=None):
    """counts: [xpos_exact, script_ok, upos_ok, n_words, uas_ok, las_ok, n_arc]."""
    model.eval()
    cnt = torch.zeros(7, dtype=torch.long, device=device)
    for i in range(0, len(dev_rows), micro):
        batch = batch_chunk(dev_rows[i:i + micro], T, W, tokenizer)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tag_out, arc, rel, wmask, sent_ids = model(batch)
        m = batch["y_script"] != -100
        if m.any():
            ok = torch.ones_like(m)
            for p, lg in enumerate(tag_out["xpos"]):
                good = lg.argmax(-1) == batch["y_xpos"][:, :, p]
                ok &= good | (batch["y_xpos"][:, :, p] == -100)
            cnt[0] += int((ok & m).sum())
            cnt[1] += int(((tag_out["script"].argmax(-1) == batch["y_script"]) & m).sum())
            cnt[2] += int(((tag_out["upos"].argmax(-1) == batch["y_upos"]) & m).sum())
            cnt[3] += int(m.sum())
        if arc is not None:
            gh, gl, gm = parse_gold(sent_ids, sentences, deprel_vocab, arc.shape[1], device)
            if decode == "mst":
                from parser.mst import mst_heads_labels
                ph, pl = mst_heads_labels(arc, rel, wmask)
            else:
                ph, pl = core.biaffine.decode(arc, rel, wmask)
            gh_c, gl_c, gm_c = gh.cpu(), gl.cpu(), gm.cpu()
            valid = (gh_c != -100) & gm_c
            cnt[4] += int(((ph == gh_c) & valid).sum())
            cnt[5] += int(((ph == gh_c) & (pl == gl_c) & valid).sum())
            cnt[6] += int(valid.sum())
    model.train()
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
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
    w_parse = cfg.get("w_parse", 1.0)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    def log0(*x):
        if rank == 0:
            print(*x, flush=True)

    kdir = Path(cfg["kfold_dir"])
    train_sents = list(read_conllu(kdir / f"train{fold}.conllu"))
    dev_sents = list(read_conllu(kdir / f"dev{fold}.conllu"))
    if cfg.get("max_sents"):
        train_sents = train_sents[:cfg["max_sents"]]
        dev_sents = dev_sents[:max(cfg["max_sents"] // 4, 8)]

    is_enc = lambda f: encode_word(f) is not None
    vocab = LabelVocab.build(train_sents, is_enc)
    deprel_vocab = DeprelVocab.build(train_sents)
    if rank == 0:
        vocab.save(out / "vocab.json")
        deprel_vocab.save(out / "deprel_vocab.json")
    log0(f"fold={fold} train={len(train_sents)} dev={len(dev_sents)} scripts={vocab.n_scripts} "
         f"tags={len(vocab.tags)} deprels={len(deprel_vocab.rels)} world={world}")

    # ---- model (loaded before the datasets since the HF path needs its tokenizer to encode)
    encoder, pcfg_pre, tokenizer = load_backbone_auto(cfg, device)
    is_hf = tokenizer is not None
    hf_max_len = cfg.get("hf_max_len", 512)
    use_cap = cfg.get("use_cap", True)
    if is_hf and use_cap:
        # see tagger/train.py for the same guard: the fine-tune-only cap channel is injected
        # into CharBERT's char-plane embeddings and has no equivalent for subword input
        log0("  [hf backbone] ignoring use_cap=true (no char-plane to inject into)")
        use_cap = False

    train_ds = TaggerDataset(train_sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=hf_max_len)
    dev_ds = TaggerDataset(dev_sents, vocab, T, W, tokenizer=tokenizer, hf_max_len=hf_max_len)
    dev_rows, _ = pack_dev_items(dev_ds.encs, W, tokenizer, T)
    dev_shard = dev_rows[rank::world]

    tcfg = TaggerConfig(pool=cfg.get("pool", "mean"), head_dropout=cfg.get("head_dropout", 0.33),
                        w_xpos=cfg.get("w_xpos", 1.0), w_script=cfg.get("w_script", 1.0),
                        w_upos=cfg.get("w_upos", 0.2), w_flat=cfg.get("w_flat", 1.0),
                        use_cap=use_cap, scalar_mix=cfg.get("scalar_mix", True))
    pcfg = ParserConfig(d_arc=cfg.get("d_arc", 500), d_rel=cfg.get("d_rel", 150),
                        dropout=cfg.get("parse_dropout", 0.33), n_labels=len(deprel_vocab.rels))
    core = JointModel(encoder, vocab, tcfg, pcfg, W=W).to(device)

    model = core
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # HF backbones (BERT/RoBERTa-style) instantiate a pooler (pooler.dense) that is never
        # called by HFBackboneWithHidden.forward() (we only consume hidden_states), so it never
        # receives a gradient -- DDP's default strict "all params must be used" check throws
        # without find_unused_parameters=True here. The CharBERT path has no such unused
        # trainable param (frozen LM heads are excluded from requires_grad entirely), so it's
        # left off there to keep that path's DDP bucketing at its original efficiency.
        model = DDP(core, device_ids=[rank % torch.cuda.device_count()],
                   find_unused_parameters=is_hf)
    # core = model.module below stays valid; frozen LM heads keep find_unused_parameters=False safe

    groups = tagger_param_groups(core.tagger, cfg)
    groups.append(dict(params=list(core.biaffine.parameters()), lr=cfg["lr_head"],
                       weight_decay=cfg.get("head_wd", 0.0), base_lr=cfg["lr_head"], is_enc=False))
    groups = [g for g in groups if any(p.requires_grad for p in g["params"])]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    ref_rows, trunc = pack_dev_items(train_ds.encs, W, tokenizer, T)
    n_rows = len(ref_rows) // world * world
    accum = int(cfg.get("grad_accum", 1))      # gradient accumulation (single-GPU runs keep the global batch)
    steps_per_epoch = max(n_rows // world // (micro * accum), 1)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(int(total_steps * cfg.get("warmup_frac", 0.1)), 1)
    log0(f"rows/epoch={n_rows} steps/epoch={steps_per_epoch} total={total_steps} "
         f"trunc={trunc} w_parse={w_parse} eff_batch_rows={micro*world}")

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        f = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    metrics_f = out / f"metrics_rank{rank}.jsonl"
    t0 = time.time()

    def do_step(batches, lr_s, step):
        for pg in opt.param_groups:
            pg["lr"] = pg["base_lr"] * lr_s
        opt.zero_grad(set_to_none=True)
        for batch in batches:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tag_out, arc, rel, wmask, sent_ids = model(batch)
                tag_loss, logs = core.tagger.loss(tag_out, batch)
                if arc is not None:
                    gh, gl, gm = parse_gold(sent_ids, train_sents, deprel_vocab, arc.shape[1], device)
                    p_loss, p_logs = core.biaffine.loss(arc, rel, gh, gl, gm)
                    loss = tag_loss + w_parse * p_loss
                else:
                    p_logs = {}
                    loss = tag_loss
            (loss / len(batches)).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        if torch.isfinite(gnorm):
            opt.step()
        if rank == 0 and step % cfg.get("log_every", 10) == 0:
            rec = dict(step=step, lr=round(opt.param_groups[0]["lr"], 7),
                       loss=round(float(loss.item()), 4), gnorm=round(float(gnorm), 3),
                       **logs, **{f"p_{k}": v for k, v in p_logs.items()},
                       sps=round(step / max(time.time() - t0, 1e-9), 2))
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as mf:
                mf.write(json.dumps(rec) + "\n")

    best_score, best_epoch, step = -1.0, -1, 0
    epoch0 = 0
    last = out / "last.pt"          # epoch-level resume
    if last.exists():
        sd = torch.load(last, map_location=device, weights_only=False)
        core.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        epoch0, step = sd["epoch"] + 1, sd["step"]
        best_score, best_epoch = sd["best_score"], sd["best_epoch"]
        log0(f"resumed from last.pt: epoch {epoch0}, step {step}, best {best_score:.4f}@{best_epoch}")
    for epoch in range(epoch0, cfg["epochs"]):
        rows, _ = pack_dev_items(train_ds.encs, W, tokenizer, T, order=torch.randperm(
            len(train_ds.encs),
            generator=torch.Generator().manual_seed(cfg.get("seed", 0) * 1000 + epoch)).tolist())
        n = len(rows) // world * world
        shard = rows[rank:n:world]
        chunk = micro * accum
        nsteps = len(shard) // chunk * chunk
        for i in range(0, nsteps, chunk):
            do_step([batch_chunk(shard[i + j:i + j + micro], T, W, tokenizer) for j in range(0, chunk, micro)],
                    lr_scale(step), step)
            step += 1

        cnt = evaluate_dev(model, core, dev_shard, dev_sents, deprel_vocab,
                           cfg.get("eval_micro", micro), device, T, W, tokenizer=tokenizer)
        if is_ddp:
            import torch.distributed as dist
            dist.all_reduce(cnt)
        nw = max(int(cnt[3]), 1); na = max(int(cnt[6]), 1)
        m = dict(epoch=epoch, xpos_exact=round(int(cnt[0]) / nw, 4),
                 lemma_script=round(int(cnt[1]) / nw, 4), upos=round(int(cnt[2]) / nw, 4),
                 uas=round(int(cnt[4]) / na, 4), las=round(int(cnt[5]) / na, 4), n_words=nw)
        score = (m["xpos_exact"] + m["lemma_script"] + m["upos"] + m["las"]) / 4
        m["score"] = round(score, 4)
        log0("  EVAL " + json.dumps(m))
        if rank == 0:
            with open(out / "eval.jsonl", "a") as ef:
                ef.write(json.dumps(m) + "\n")
            if score > best_score:
                tmp = out / "best.pt.tmp"
                torch.save(dict(model=core.state_dict(), tcfg=vars(tcfg), pcfg=vars(pcfg),
                                cfg=cfg, pretrain_cfg=pcfg_pre, epoch=epoch, dev=m, W=W, T=T,
                                deprel_vocab=deprel_vocab.rels), tmp)
                os.replace(tmp, out / "best.pt")
                print(f"  new best.pt (score {score:.4f} las {m['las']:.4f})", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
        if rank == 0:
            tmp = out / "last.pt.tmp"
            torch.save(dict(model=core.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step,
                            best_score=best_score, best_epoch=best_epoch), tmp)
            os.replace(tmp, last)
        if epoch - best_epoch >= cfg.get("patience", 8):
            log0(f"EARLY STOP epoch={epoch} best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    log0(f"DONE best_score={best_score:.4f} best_epoch={best_epoch} ({(time.time()-t0)/60:.1f} min)")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
