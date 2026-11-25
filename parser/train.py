"""Train a deep-biaffine dependency parser over the Stoicheia char arm with a learned
scalar mix. With finetune=true the backbone trains too (enc_lr) alongside the scalar-mix +
biaffine head (lr), like morphparse.

Single-process OR torchrun/DDP (data-parallel over sentences; the treebank is small, every
rank holds it all in RAM and takes an equal disjoint slice of the shuffled sentences each
epoch — equal slice sizes keep step counts identical across ranks so the all-reduce never
deadlocks). Dev/test LAS/UAS are computed in-house over the ENCODABLE (Greek-letter-bearing)
token subset only.

  python -m parser.train --config configs/parser_char.json                 # 1 GPU
  torchrun --nproc_per_node=4 -m parser.train --config configs/parser_char.json   # 4 GPU

NOTE: this release only supports arm="char". The original 3-arm ablation (char / lemma /
fused) also trained "lemma" and "fused" arms over a second, LemmaDiff-grc encoder; that
encoder is a separate, unpublished side-repo, so those two arms — and the build_lemma_arm
loader that fed them — were dropped here (see parser/model.py's module docstring). The
published joint model (parser/joint_train.py) supersedes all three of these single-arm
specialists on test LAS, so this standalone char-arm trainer is kept mainly for reference /
ablation reproduction rather than as the recommended training path.
"""
from __future__ import annotations

import argparse, datetime, json, math, os, time
from pathlib import Path

import torch
import torch.distributed as dist

from tagger.backbone import load_backbone
from tagger.conllu import read_conllu

from parser.biaffine import ParserConfig, BiaffineHead
from parser.labels import DeprelVocab
from parser.model import CharArm, SyntaxModel


def build_char_arm(device, attn="sdpa", finetune=False):
    """Load the frozen Stoicheia backbone (formerly parser/build.py, trimmed of the
    LemmaDiff-grc-dependent build_lemma_arm sibling that lived alongside it)."""
    ckpt = os.path.expandvars(os.environ["STOICHEIA_CKPT"])
    model, pcfg = load_backbone(ckpt, device, attn_impl=attn)
    n_layers = pcfg["depth"] + 1
    return CharArm(model, n_layers, finetune=finetune).to(device), pcfg["d_model"]


def ddp_setup():
    """-> (rank, world, local_rank, is_ddp). No-op (0,1,0,False) outside torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        rank = dist.get_rank(); world = dist.get_world_size()
        local = rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        return rank, world, local, True
    return 0, 1, 0, False


@torch.no_grad()
def eval_counts(core, sents, deprel_vocab, T, W, device, micro, decode="greedy"):
    """UAS/LAS *counts* (not ratios) over `sents` — caller reduces across ranks then divides.
    decode='mst' uses Chu-Liu-Edmonds (valid-tree) instead of greedy per-token argmax."""
    core.eval()
    uas_c = las_c = n_c = 0
    for i in range(0, len(sents), micro):
        batch = sents[i:i + micro]
        arc_scores, rel_scores, heads, labels, mask = core(batch, deprel_vocab, T, W, device)
        if arc_scores is None:
            continue
        if decode == "mst":
            from parser.mst import mst_heads_labels
            pred_heads, pred_labels = mst_heads_labels(arc_scores, rel_scores, mask)
        else:
            pred_heads, pred_labels = core.head.decode(arc_scores, rel_scores, mask)
        gh, gl, m = heads.cpu(), labels.cpu(), mask.cpu()
        valid = (gh != -100) & m
        uas_c += int(((pred_heads == gh) & valid).sum())
        las_c += int(((pred_heads == gh) & (pred_labels == gl) & valid).sum())
        n_c += int(valid.sum())
    core.train()
    return uas_c, las_c, n_c


@torch.no_grad()
def evaluate(model, sents, deprel_vocab, T, W, device, micro, decode="greedy"):
    """Ratio form for single-process eval (parser/evaluate.py on the test split)."""
    uc, lc, nc = eval_counts(model, sents, deprel_vocab, T, W, device, micro, decode)
    return uc / max(nc, 1), lc / max(nc, 1), nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    for k in ("out_dir", "kfold_dir"):
        cfg[k] = os.path.expandvars(cfg[k])

    rank, world, local, is_ddp = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))
    out = Path(cfg["out_dir"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if is_ddp:
        dist.barrier()

    def log0(*args, **kw):
        if rank == 0:
            print(*args, **kw, flush=True)

    arm = cfg["arm"]
    if arm != "char":
        raise NotImplementedError(
            f'arm={arm!r} not supported in this release — only "char" is (the "lemma" and '
            '"fused" arms depended on the unpublished LemmaDiff-grc side-repo; see parser/model.py).')
    kdir = Path(cfg["kfold_dir"]); fold = cfg.get("fold", 0)
    train_sents = list(read_conllu(kdir / f"train{fold}.conllu"))
    dev_sents = list(read_conllu(kdir / f"dev{fold}.conllu"))
    if cfg.get("max_sents"):
        train_sents = train_sents[:cfg["max_sents"]]
        dev_sents = dev_sents[:max(cfg["max_sents"] // 4, 8)]

    deprel_vocab = DeprelVocab.build(train_sents)
    if rank == 0:
        deprel_vocab.save(out / "deprel_vocab.json")
    log0(f"arm={arm} train={len(train_sents)} dev={len(dev_sents)} "
         f"n_labels={len(deprel_vocab.rels)} world={world}")

    attn = cfg.get("attn", "sdpa")
    ft = cfg.get("finetune", False)
    lemma_arm = None       # always None in this release (see arm check above)
    char_arm, d_in = build_char_arm(device, attn, finetune=ft)
    log0(f"char arm loaded, d={d_in}")

    pcfg = ParserConfig(d_arc=cfg.get("d_arc", 500), d_rel=cfg.get("d_rel", 150),
                       dropout=cfg.get("dropout", 0.33), n_labels=len(deprel_vocab.rels))
    head = BiaffineHead(d_in, pcfg).to(device)
    core = SyntaxModel(arm, char_arm, lemma_arm, head).to(device)

    # two LR groups: head + scalar-mix (fast) vs fine-tuned backbone (slow), like morphparse.
    head_mix = list(head.parameters())
    if char_arm is not None:
        head_mix += list(char_arm.mix.parameters())
    if lemma_arm is not None:
        head_mix += list(lemma_arm.mix.parameters())
    enc_params = []
    if ft:
        if char_arm is not None:
            enc_params += [p for p in char_arm.model.parameters() if p.requires_grad]
        if lemma_arm is not None:
            enc_params += [p for p in lemma_arm.model.parameters() if p.requires_grad]
    lr_head = cfg.get("lr", 8e-4); lr_enc = cfg.get("enc_lr", 2e-5)
    groups = [{"params": head_mix, "lr": lr_head, "base": lr_head}]
    if enc_params:
        groups.append({"params": enc_params, "lr": lr_enc, "base": lr_enc})
    trainable = head_mix + enc_params
    log0(f"trainable params: {sum(p.numel() for p in trainable)} "
         f"(head+mix {sum(p.numel() for p in head_mix)}, enc {sum(p.numel() for p in enc_params)}, "
         f"finetune={ft})")
    opt = torch.optim.AdamW(groups, weight_decay=cfg.get("wd", 0.0))

    # Wrap for DDP AFTER building the optimizer on the raw params (DDP only adds grad hooks,
    # it does not replace the param tensors). find_unused_parameters=True: the backbone's
    # LM/unembedding head is never touched by feature extraction (return_layers), so a handful
    # of its params get no grad each step — harmless (AdamW skips grad=None), but DDP must be
    # told to expect it or it raises "param did not receive grad".
    model = core
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(core, device_ids=[local], find_unused_parameters=True)

    T, W = cfg.get("T", 2048), cfg.get("W", 384)
    micro = cfg.get("micro_batch", 16)
    epochs = cfg["epochs"]
    per_rank = max((len(train_sents) // world // micro) * micro, micro)   # equal shard/rank
    steps_per_epoch = per_rank // micro
    total_steps = steps_per_epoch * epochs
    warmup = max(int(total_steps * cfg.get("warmup_frac", 0.1)), 1)
    log0(f"per_rank={per_rank} steps/epoch={steps_per_epoch} total_steps={total_steps} "
         f"eff_batch={micro * world}")

    def lr_scale(step):
        if step < warmup:
            return step / warmup
        f = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(f, 1.0)))

    metrics_f = out / "metrics.jsonl"
    best_las, best_epoch, step = -1.0, -1, 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_sents), generator=torch.Generator().manual_seed(
            cfg.get("seed", 0) * 1000 + epoch)).tolist()                  # SAME order on every rank
        shuffled = [train_sents[i] for i in order]
        shard = shuffled[rank * per_rank:(rank + 1) * per_rank]           # disjoint, equal length
        for i in range(0, per_rank, micro):
            batch_sents = shard[i:i + micro]
            arc_scores, rel_scores, heads, labels, mask = model(batch_sents, deprel_vocab, T, W, device)
            if arc_scores is None:
                continue          # unreachable for real Greek batches (every sentence is encodable)
            loss, logs = core.head.loss(arc_scores, rel_scores, heads, labels, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.get("clip", 5.0))
            for pg in opt.param_groups:
                pg["lr"] = pg["base"] * lr_scale(step)                    # per-group base LR (enc vs head)
            if torch.isfinite(gnorm):
                opt.step()
            if rank == 0 and step % cfg.get("log_every", 20) == 0:
                rec = dict(epoch=epoch, step=step, loss=round(loss.item(), 4),
                          gnorm=round(float(gnorm), 3), **logs)
                print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
                with open(metrics_f, "a") as mf:
                    mf.write(json.dumps(rec) + "\n")
            step += 1

        # dev eval: each rank its shard, reduce counts -> identical LAS everywhere -> consistent stop
        uc, lc, nc = eval_counts(core, dev_sents[rank::world], deprel_vocab, T, W, device,
                                 cfg.get("eval_micro", micro))
        if is_ddp:
            t = torch.tensor([uc, lc, nc], dtype=torch.long, device=device)
            dist.all_reduce(t)
            uc, lc, nc = t.tolist()
        uas, las = uc / max(nc, 1), lc / max(nc, 1)
        log0(f"  EVAL epoch={epoch} uas={uas:.4f} las={las:.4f} n={nc}")
        if rank == 0:
            with open(out / "eval.jsonl", "a") as ef:
                ef.write(json.dumps(dict(epoch=epoch, uas=uas, las=las, n=nc)) + "\n")
        if las > best_las:
            best_las, best_epoch = las, epoch
            if rank == 0:
                state = dict(head=head.state_dict(), arm=arm, cfg=cfg, epoch=epoch, uas=uas, las=las,
                            deprel_vocab=deprel_vocab.rels, d_in=d_in, finetune=ft)
                if char_arm is not None:
                    state["char_mix"] = char_arm.mix.state_dict()
                    if ft:                               # persist fine-tuned backbone, else eval reloads the ORIGINAL
                        state["char_backbone"] = char_arm.model.state_dict()
                if lemma_arm is not None:
                    state["lemma_mix"] = lemma_arm.mix.state_dict()
                    if ft:
                        state["lemma_backbone"] = lemma_arm.model.state_dict()
                tmp = out / "best.pt.tmp"
                torch.save(state, tmp); os.replace(tmp, out / "best.pt")
                log0(f"  new best.pt (las={las:.4f})")
        if epoch - best_epoch >= cfg.get("patience", 8):
            log0(f"EARLY STOP epoch={epoch} best_epoch={best_epoch} best_las={best_las:.4f}")
            break
    log0(f"DONE arm={arm} best_las={best_las:.4f} best_epoch={best_epoch} "
         f"({(time.time()-t0)/60:.1f} min)")
    if is_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
