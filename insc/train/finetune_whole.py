"""Whole-document (lacuna-preserving) documentary restoration finetune. Loads full PHI
inscriptions + papyri via iphi.py's/papyri.py's load_whole_full() -- documents are NEVER
split at a real lacuna; the true '-'/'...' gaps stay in place, always fed as masked input,
never a supervision target (see train/noising.py's is_real_lacuna support). Optional region/
century metadata conditioning with per-document dropout (p_region_none/p_century_none) so
the model works whether or not a real fragment's provenance/date is known. In-memory (the
whole PHI+papyri corpus is ~60M characters, far below GB-scale shard territory) -- no
TierSpec/DataConfig/shard machinery needed.

Dev-stall-driven schedule: warmup -> constant peak LR until dev loss
stalls -> cosine decay -> early stop, best.pt is the model of record. When meta_condition is
on, the dev fixture is scored TWICE per eval -- once with real region/century, once with both
forced to UNK -- so the with-vs-without-metadata comparison is visible throughout training,
not just at the end.

  python insc/train/finetune_whole.py --config configs/insc/finetune_whole_t0v1.json
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "data"))
from model.char_bert import CharBertConfig, CharBertEncoder, num_params
from meta_vocab import N_REGION, N_CENTURY, UNK_REGION, UNK_CENTURY
from train.loss import compute_loss
from train.noising import NoiseConfig
from train.collate import pack_batch, collate, UNK_BND, UNK_DIA, UNK_PUNCT
from train.train import ddp_setup, save_ckpt


def lr_mult_dyn(step, warmup, anneal_start, decay_len):
    """Warmup -> flat peak until anneal_start -> cosine to 0.02 over decay_len."""
    if step < warmup:
        return step / max(warmup, 1)
    if anneal_start is None or step < anneal_start:
        return 1.0
    t = min((step - anneal_start) / max(decay_len, 1), 1.0)
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))


def stalled(series, window=8, eps=0.002):
    if len(series) < window:
        return False
    half = window // 2
    return min(series[-half:]) > min(series[-window:-half]) - eps


def load_literary(n_records, min_len, max_len, seed=0):
    """Whole literary documents from the doc_clean pretraining corpus, in the same record
    shape as load_whole_full(). Literary text carries no lacunae (is_real_lacuna all False)
    and no provenance metadata (region/century UNK). Present purely as an anti-forgetting
    / regularisation tier, mirroring the ~11% literary (gold+silver) replay share the
    shard-based fine-tune recipe uses."""
    from data.normalize import Stats, normalize_record
    # Reads a PRE-EXTRACTED plain-JSONL sample (see raw/literary_sample.jsonl, built once
    # by sampling every 17th qualifying record across the whole doc_clean corpus so all
    # nine source tiers are represented, not just the `bronze` prefix). Plain JSONL on
    # purpose: no zstandard/zstd dependency at training time, and identical bytes every run.
    src = os.path.expandvars("$INS_DATA/raw/literary_sample.jsonl")
    st = Stats(); out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            if len(out) >= n_records:
                break
            try:
                t = (json.loads(line).get("text") or "").strip()
            except Exception:
                continue
            if not (min_len <= len(t) <= max_len):
                continue
            nr = normalize_record(t, st, with_punct=True)
            if nr is None:
                continue
            chars, boundary, dia, cap, punct = nr
            n = len(chars)
            if n < min_len:
                continue
            out.append(dict(chars=chars, boundary=boundary, dia=dia, cap=cap, punct=punct,
                            is_real_lacuna=np.zeros(n, dtype=bool),
                            phi_id=f"lit{len(out)}", seg=0, split="train",
                            region_id=UNK_REGION, century_id=UNK_CENTURY))
    return out


def load_domain_records(split, min_len, mix=None, seq_len=4096, seed=0):
    """Whole-document records per tier -- NEVER split at a real lacuna. Returns
    (pools, weights, names). Tiers: iphi (real editions), iphi_syn (synthetic editions of
    the same inscriptions -- augmentation), papyri, literary (doc_clean sample). Weights
    come from the config `mix`; a tier with weight 0 is not even loaded."""
    from iphi import load_whole_full as iphi_load
    from papyri import load_whole_full as pap_load
    mix = mix or {}
    w_iphi = float(mix.get("iphi", 1.0))
    w_syn = float(mix.get("iphi_syn", 0.0))
    w_pap = float(mix.get("papyri", 1.0))
    w_lit = float(mix.get("literary", 0.0))
    pools, weights, names = [], [], []
    if w_iphi > 0:
        pools.append(iphi_load(split=split, min_len=min_len)); weights.append(w_iphi)
        names.append("iphi")
    if w_syn > 0:
        pools.append(iphi_load(split=split, min_len=min_len, field="synthetic"))
        weights.append(w_syn); names.append("iphi_syn")
    if w_pap > 0:
        pools.append(pap_load(split=split, min_len=min_len)); weights.append(w_pap)
        names.append("papyri")
    # CLEAN-SEGMENT tiers: iphi.load()/papyri.load() split each text AT every real lacuna,
    # giving lacuna-free contiguous segments. That is exactly what the shard-based
    # `both_ft_docclean` run trained on, and why it still wins the clean-context and STRICT
    # protocols -- whole-document runs never see that distribution. Carrying both tiers
    # trains ONE model for both deployment shapes (whole edited document with surviving
    # lacunae, and clean excerpt) instead of trading one against the other.
    w_iseg = float(mix.get("iphi_seg", 0.0))
    w_pseg = float(mix.get("pap_seg", 0.0))
    if w_iseg > 0:
        from iphi import load as iphi_seg_load
        segs = iphi_seg_load(split=split, min_len=min_len)
        for r in segs:
            r.setdefault("is_real_lacuna", np.zeros(len(r["chars"]), dtype=bool))
        pools.append(segs); weights.append(w_iseg); names.append("iphi_seg")
    if w_pseg > 0:
        from papyri import load as pap_seg_load
        segs = pap_seg_load(split=split, min_len=min_len)
        for r in segs:
            r.setdefault("is_real_lacuna", np.zeros(len(r["chars"]), dtype=bool))
        pools.append(segs); weights.append(w_pseg); names.append("pap_seg")
    if w_lit > 0 and split == "train":       # literary is a TRAIN-only regularisation tier
        pools.append(load_literary(int(mix.get("literary_n", 60000)), min_len, seq_len, seed))
        weights.append(w_lit); names.append("literary")
    # Config weights are CHARACTER-share targets, but infinite_mix samples RECORDS.
    # Tier mean lengths differ ~9x (iphi_seg ~106 chars, literary ~956), so using the
    # weights as record probabilities silently hands the long-record tiers several times
    # their intended share of the training signal (measured: literary 0.3 -> 24% of
    # characters, papyri 47%). Convert: record_weight = char_share / mean_record_len.
    weights = [w / max(float(np.mean([len(r["chars"]) for r in p_])), 1.0)
               for w, p_ in zip(weights, pools)]
    return pools, weights, names


def infinite_mix(pools, weights, rng):
    """Infinite generator over N record pools at fixed weight ratio; each pool is
    internally shuffled and re-permuted once exhausted. Generalised from the original
    two-pool (iphi/papyri) version so the finetune mix can also carry synthetic
    inscription editions and literary Greek -- the two ingredients the shard-based
    `both_ft_docclean` run had (iphi_syn 0.5, gold+silver 0.30) and the whole-document
    pipeline was missing."""
    keep = [(p, w) for p, w in zip(pools, weights) if p and w > 0]
    assert keep, "no non-empty pool with positive weight"
    pools = [p for p, _ in keep]
    weights = [w for _, w in keep]
    w = np.array(weights, dtype=float); w = w / w.sum()
    idxs = [0] * len(pools)
    perms = [rng.permutation(len(p)) for p in pools]
    while True:
        pi = int(rng.choice(len(pools), p=w))
        if idxs[pi] >= len(perms[pi]):
            perms[pi] = rng.permutation(len(pools[pi])); idxs[pi] = 0
        rec = pools[pi][int(perms[pi][idxs[pi]])]
        idxs[pi] += 1
        yield rec


def dev_fixture_records(recs, exclude_path, n, seed=1234):
    """Fixed sample of clean val records (contamination-excluded), same convention as
    the documentary dev-record builder."""
    excl = set()
    if exclude_path:
        j = json.loads(Path(exclude_path).read_text())
        excl = {(str(x[0]), int(x[1])) for x in j["contaminated"]}
    filtered = [r for r in recs if 64 <= len(r["chars"]) <= 1500
                and (str(r["phi_id"]), int(r.get("seg", 0))) not in excl]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(filtered), size=min(n, len(filtered)), replace=False)
    return [filtered[i] for i in pick]


def force_unk_metadata(batch):
    """Clone of a packed batch with region/century forced to UNK everywhere -- the
    "metadata withheld" condition, for the with-vs-without comparison and for the
    Ithaca-comparable eval (Ithaca has no metadata-conditioning capability at all)."""
    b = dict(batch)
    b["region"] = torch.full_like(batch["region"], UNK_REGION)
    b["century"] = torch.full_like(batch["century"], UNK_CENTURY)
    return b


@torch.no_grad()
def dev_bits_per_char(model, batch, device, micro=32):
    """Mean bits/char over the dev batch, forwarded in row-chunks: one monolithic forward
    of all 256 rows fits at T=4096 but OOMs at T=8192. Sum-CE aggregation keeps the result
    bit-identical to the single-forward version."""
    import torch.nn.functional as F
    n_rows = batch["input_ids"].shape[0]
    tot_ce, tot_n = 0.0, 0
    for i in range(0, n_rows, micro):
        b = {k: (v[i:i + micro].to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
        out = model(b)
        lab = b["labels"]
        m = lab != -100
        if not m.any():
            continue
        tot_ce += float(F.cross_entropy(out["char"][m], lab[m], reduction="sum").item())
        tot_n += int(m.sum())
    if tot_n == 0:
        return float("nan")
    return tot_ce / tot_n / math.log(2)


def _levenshtein(a, b):
    if not len(a): return len(b)
    if not len(b): return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def build_dev_cer_fixture(recs, T, n_per_L=20, L_max=20, seed=1234, micro=16):
    """The realistic dev task: whole documents (real lacunae intact as MASK context), ONE
    synthetic span of length L=1..L_max placed on fully-known positions, greedy restore,
    CER vs gold. Returns pre-built (batch_dict, meta) micro-batches, fixed once so the dev
    series is comparable step to step. Mirrors insc/eval/restore.py::eval_span_whole's
    placement rules exactly (never on a real lacuna, whole doc as context)."""
    rng = np.random.default_rng(seed)
    samples = []
    for L in range(1, L_max + 1):
        cands = [r for r in recs if L + 8 < len(r["chars"]) <= T]
        rng.shuffle(cands)
        made = 0
        for r in cands:
            if made >= n_per_L:
                break
            knownable = ~np.asarray(r["is_real_lacuna"], dtype=bool)
            valid = [s for s in range(4, len(r["chars"]) - L - 4)
                     if knownable[s:s + L].all()]
            if not valid:
                continue
            s = int(rng.choice(valid))
            samples.append((r, s, L))
            made += 1
    batches = []
    for i in range(0, len(samples), micro):
        chunk = samples[i:i + micro]
        B = len(chunk)
        ids = torch.full((B, T), 26, dtype=torch.long)       # pad_id (NoiseConfig.pad_id)
        bnd = torch.full((B, T), UNK_BND, dtype=torch.long)
        dia = torch.full((B, T), UNK_DIA, dtype=torch.long)
        punct = torch.full((B, T), UNK_PUNCT, dtype=torch.long)
        region = torch.full((B, T), UNK_REGION, dtype=torch.long)
        century = torch.full((B, T), UNK_CENTURY, dtype=torch.long)
        seg = torch.zeros(B, T, dtype=torch.long)
        meta = []
        for b, (r, s, L) in enumerate(chunk):
            n = len(r["chars"])
            chars = np.asarray(r["chars"], np.int64).copy()
            gold = chars[s:s + L].copy()
            chars[s:s + L] = 24                              # mask_id (NoiseConfig.mask_id)
            brow = np.minimum(np.asarray(r["boundary"], np.int64), 2)
            brow[s:s + L] = UNK_BND
            ids[b, :n] = torch.from_numpy(chars)
            bnd[b, :n] = torch.from_numpy(brow)
            region[b, :n] = int(r.get("region_id", UNK_REGION))
            century[b, :n] = int(r.get("century_id", UNK_CENTURY))
            seg[b, :n] = 1
            meta.append((s, L, gold))
        batches.append((dict(input_ids=ids, boundary=bnd, dia=dia, punct=punct,
                             region=region, century=century, seg_id=seg), meta))
    return batches


@torch.no_grad()
def dev_span_cer(model, fixture, device):
    """Greedy (argmax) restore of every fixture span -> mean CER + exact-match rate,
    plus CER by length band. Cheap: one forward per micro-batch, no beam."""
    cers, exact = [], []
    band = {"1-5": [], "6-10": [], "11-20": []}
    for batch, meta in fixture:
        b = {k: v.to(device) for k, v in batch.items()}
        out = model(b)
        pred = out["char"][:, :, :24].argmax(-1).cpu().numpy()
        for i, (s, L, gold) in enumerate(meta):
            p = pred[i, s:s + L]
            c = _levenshtein(list(p), list(gold)) / max(L, 1)
            cers.append(c)
            exact.append(int((p == gold).all()))
            key = "1-5" if L <= 5 else ("6-10" if L <= 10 else "11-20")
            band[key].append(c)
    return dict(cer=float(np.mean(cers)), exact=float(np.mean(exact)),
                **{f"cer_{k}": round(float(np.mean(v)), 4) for k, v in band.items() if v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg["out_dir"] = os.path.expandvars(cfg["out_dir"])
    # the config-pinned test/val digit is authoritative: it is exported before any data loads
    if "test_digit" in cfg:
        os.environ["INSC_TEST_DIGIT"] = str(cfg["test_digit"])
    if "val_digit" in cfg:
        os.environ["INSC_VAL_DIGIT"] = str(cfg["val_digit"])
    rank, world, is_ddp = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"] + rank)
    torch.set_float32_matmul_precision("high")

    T = cfg["seq_len"]; rows = cfg["micro_batch"]; accum = cfg.get("grad_accum", 1)
    hard_max = cfg.get("total_steps", 12000)
    decay_len = cfg.get("decay_len", 1500)
    warmup = cfg.get("warmup", 200)
    eval_every = cfg.get("eval_every", 250)
    patience = cfg.get("anneal_patience", 3)
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    meta_condition = cfg.get("meta_condition", False)
    mcfg = CharBertConfig(attn_impl=cfg.get("attn", "flex"), d_model=cfg["d_model"],
                          n_heads=cfg["d_model"] // 64, depth=cfg["depth"],
                          char_window=cfg["char_window"], qk_norm=True,
                          n_region=N_REGION if meta_condition else 0,
                          n_century=N_CENTURY if meta_condition else 0)
    model = CharBertEncoder(mcfg).to(device)

    step0 = 0
    ckpt = out / "last.pt"
    sd = None
    if ckpt.exists():
        sd = torch.load(ckpt, map_location=device)
        model.load_state_dict(sd["model"]); step0 = sd["step"]
        if rank == 0:
            print(f"resumed finetune from step {step0}")
    elif cfg["torso"] == "random":
        if rank == 0:
            print("RANDOM INIT (smoke test / ablation): no pretrained torso loaded")
    else:
        torso = os.path.expandvars(cfg["torso"])
        tsd = torch.load(torso, map_location=device)
        res = model.load_state_dict(tsd["model"], strict=not meta_condition)
        if meta_condition:
            assert set(res.missing_keys) <= {"e_region.weight", "e_century.weight"}, res.missing_keys
            assert not res.unexpected_keys, res.unexpected_keys
        if rank == 0:
            print(f"initialized from torso {torso} (pretrain step {tsd.get('step')})"
                  + (" [meta_condition: region/century embeddings randomly init'd]"
                     if meta_condition else ""))
    if rank == 0:
        print(f"params={num_params(model)/1e6:.1f}M T={T} rows={rows}x{accum} world={world} "
              f"hard_max={hard_max} decay_len={decay_len} meta_condition={meta_condition}")
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank % torch.cuda.device_count()])
    fwd = torch.compile(model) if cfg.get("compile", True) and device.type == "cuda" else model

    decay_p = [p for p in model.parameters() if p.ndim >= 2]
    nodecay_p = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": cfg.get("wd", 0.1)},
                             {"params": nodecay_p, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.95), fused=(device.type == "cuda"))
    if sd is not None and "opt" in sd:
        try:
            opt.load_state_dict(sd["opt"])
        except Exception:
            pass

    nc = cfg.get("noise", {})
    ncfg = NoiseConfig(w_span=nc.get("w_span", 0.45), w_word=nc.get("w_word", 0.15),
                       w_elastic=nc.get("w_elastic", 0.15), w_iid=nc.get("w_iid", 0.15),
                       w_halfword=nc.get("w_halfword", 0.15),
                       w_substitute=nc.get("w_substitute", 0.10),
                       span_mean=nc.get("span_mean", 4.0), span_max=nc.get("span_max", 12),
                       p_region_none=(nc.get("p_region_none", 0.3) if meta_condition else 0.0),
                       p_century_none=(nc.get("p_century_none", 0.3) if meta_condition else 0.0))

    min_len = cfg.get("min_len", 20)
    mix = cfg.get("mix", {"iphi": 1.0, "papyri": 1.0})
    pools, weights, names = load_domain_records("train", min_len, mix=mix,
                                                seq_len=T, seed=cfg["seed"])
    if rank == 0:
        print("train tiers: " + "  ".join(f"{n}={len(p)}@w{w:g}"
              for n, p, w in zip(names, pools, weights)), flush=True)
    rng = np.random.default_rng(cfg["seed"] + step0 + rank * 1000003)
    it = infinite_mix(pools, weights, rng)
    g = torch.Generator().manual_seed(cfg["seed"] + step0 + rank * 1000003)

    # optionally build packed batches in worker processes (pack_batch is a per-record Python loop,
    # which bounds the step rate of a single-GPU run)
    nworkers = int(cfg.get("num_workers", 0))
    batch_it = None
    if nworkers > 0:
        class _Batches(torch.utils.data.IterableDataset):
            def __iter__(self_inner):
                info = torch.utils.data.get_worker_info()
                wid = info.id if info else 0
                seed = cfg["seed"] + step0 * 7919 + rank * 1000003 + wid * 104729
                w_it = infinite_mix(pools, weights, np.random.default_rng(seed))
                w_g = torch.Generator().manual_seed(seed)
                while True:
                    yield pack_batch(w_it, ncfg, T, rows, w_g)
        batch_it = iter(torch.utils.data.DataLoader(_Batches(), batch_size=None, num_workers=nworkers,
                                                    prefetch_factor=4, persistent_workers=True, pin_memory=True))

    dev_batch = dev_batch_unk = dev_cer_fixture = None
    if rank == 0:
        # dev fixtures are built from the REAL iphi + papyri val tiers only, selected BY
        # NAME -- positional indexing broke silently when the mix grew extra tiers
        # (vpools[-1] became pap_seg, changing the fixture composition vs earlier runs).
        vpools, _, vnames = load_domain_records("val", min_len, mix=mix, seq_len=T)
        vby = dict(zip(vnames, vpools))
        iphi_val = vby.get("iphi", [])
        pap_val = vby.get("papyri", [])
        excl = os.path.expandvars(cfg.get("dev_exclude", "")) or None
        recs = dev_fixture_records(iphi_val + pap_val, excl, cfg.get("eval_n", 256))
        print(f"dev fixture: {len(recs)} clean val records "
              f"({sum(1 for r in recs if r['is_real_lacuna'].any())} with >=1 real lacuna)")
        g_dev = torch.Generator().manual_seed(1234)
        dev_batch = collate(recs, ncfg, T, g_dev)
        dev_batch_unk = force_unk_metadata(dev_batch) if meta_condition else None
        # the realistic dev task drives the schedule: greedy span-restoration CER over
        # L=1..20 on whole documents (real lacunae in context), fixed spans -> comparable
        # step to step. bpc is still logged, but stall/anneal/best follow CER.
        all_val = [r for r in iphi_val + pap_val if 64 <= len(r["chars"]) <= T]
        dev_cer_fixture = build_dev_cer_fixture(
            all_val, T, n_per_L=cfg.get("dev_cer_n_per_L", 20),
            L_max=cfg.get("dev_cer_L_max", 20))
        n_spans = sum(len(m) for _, m in dev_cer_fixture)
        print(f"dev CER fixture: {n_spans} spans (L=1..{cfg.get('dev_cer_L_max', 20)}, "
              f"whole-document context)")

    marker = out / "anneal_start.json"
    anneal_start = None
    if cfg.get("fixed_anneal_start") is not None and not marker.exists():
        # ablations: fixed, compute-matched schedule for every arm (no dev-stall anneal)
        marker.write_text(json.dumps(dict(step=int(cfg["fixed_anneal_start"]), reason="fixed")))
    if marker.exists():
        anneal_start = json.loads(marker.read_text())["step"]
        if rank == 0:
            print(f"anneal marker: from step {anneal_start}")

    model.train()
    t0 = time.time(); seen = 0
    metrics_f = out / "metrics.jsonl"
    dev_series = []      # (step, bpc) -- drives stall/anneal
    cer_series = []      # (step, dev CER) -- drives best.pt selection
    stop = False
    step = step0 - 1      # a resume at the final step leaves the loop body unexecuted
    for step in range(step0, hard_max):
        if anneal_start is not None and step >= anneal_start + decay_len:
            break
        lr = cfg["lr"] * lr_mult_dyn(step, warmup, anneal_start, decay_len)
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = next(batch_it) if batch_it is not None else pack_batch(it, ncfg, T, rows, g)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
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
                       chps=round(seen / (time.time() - t0) / 1e6, 3),
                       anneal=anneal_start is not None, **logs)
            print("  " + " ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
            with open(metrics_f, "a") as f:
                f.write(json.dumps(rec) + "\n")

        # Eval-free periodic checkpoint (cfg "save_every", 0=off), for runs that
        # disable the dev-eval block: without it such a run's only checkpoint is
        # final.pt, so any interruption costs the entire run.
        save_every = int(cfg.get("save_every", 0) or 0)
        if save_every and step > step0 and step % save_every == 0 \
                and step % eval_every != 0:
            # ALL ranks enter this block: rank 0 saves while the others park in
            # a barrier. A rank-0-only save with the other ranks free to run into
            # the next backward allreduce can deadlock on shared filesystems; the
            # barrier mirrors the eval block's (working) synchronization. Do not
            # remove it.
            if rank == 0:
                save_ckpt(dict(model=(model.module if is_ddp else model).state_dict(),
                               opt=opt.state_dict(), step=step + 1, cfg=cfg), ckpt)
                print(f"  ckpt saved at step {step} (save_every)", flush=True)
            if is_ddp:
                import torch.distributed as dist
                dist.barrier()

        if step > step0 and step % eval_every == 0:
            if rank == 0:
                # Rank 0 checkpoints + evaluates alone while every other rank waits
                # at the broadcast below; the NCCL process-group timeout
                # (train/train.py::ddp_setup) bounds how long they will wait. Do not
                # guard this block with signal.alarm(): SIGALRM interrupts the
                # checkpoint write syscall and torch.save fails with EINTR. The
                # try/except keeps a failing eval from killing training.
                try:
                    core = model.module if is_ddp else model
                    save_ckpt(dict(model=core.state_dict(), opt=opt.state_dict(),
                                   step=step + 1, cfg=cfg), ckpt)
                    bpc = dev_bits_per_char(core, dev_batch, device)
                    m = dict(step=step, bits_per_char=bpc)
                    if dev_batch_unk is not None:
                        m["bits_per_char_unk_meta"] = dev_bits_per_char(core, dev_batch_unk, device)
                    cer_m = dev_span_cer(core, dev_cer_fixture, device)
                    m["dev_cer"] = round(cer_m["cer"], 4)
                    m["dev_exact"] = round(cer_m["exact"], 4)
                    m.update({k: v for k, v in cer_m.items() if k.startswith("cer_")})
                    print("  DEV " + json.dumps(m), flush=True)
                    with open(out / "eval.jsonl", "a") as f:
                        f.write(json.dumps(m) + "\n")
                    # Two separable decisions, deliberately driven by different signals:
                    #  * stall/anneal detection uses bits_per_char -- smooth and monotone,
                    #    so the detector is not fooled by metric noise (greedy span CER
                    #    has ~0.007 sd run-to-run, which swamps stall_eps=0.002).
                    #  * best.pt is selected on dev CER -- the target metric. bpc and CER
                    #    decorrelate late in training: bpc can bottom out and rise while
                    #    CER is still improving.
                    dev_series.append((step, bpc))          # bpc drives stalled() below
                    cer_series.append((step, cer_m["cer"]))
                    if cer_m["cer"] <= min(c for _, c in cer_series):
                        import shutil
                        shutil.copyfile(ckpt, out / "best.pt")
                        print(f"  new best.pt (dev CER {cer_m['cer']:.4f}, dev bpc {bpc:.4f})",
                              flush=True)
                    series = [b for _, b in dev_series]
                    if anneal_start is None and step >= warmup + eval_every * cfg.get("stall_window", 8) \
                            and stalled(series, cfg.get("stall_window", 8), cfg.get("stall_eps", 0.002)):
                        anneal_start = step + 1
                        marker.write_text(json.dumps(dict(step=anneal_start, reason="dev-stall")))
                        print(f"  DEV-STALL: annealing from {anneal_start}, ends "
                              f"{anneal_start + decay_len}", flush=True)
                    if anneal_start is not None and patience > 0 \
                            and step >= anneal_start + decay_len // 2:
                        ann = [(s, b) for s, b in dev_series if s > anneal_start]
                        if len(ann) > patience:
                            bi = min(range(len(ann)), key=lambda i: ann[i][1])
                            if len(ann) - 1 - bi >= patience:
                                stop = True
                                print(f"  DEV EARLY-STOP (best at step {ann[bi][0]}); best.pt "
                                      f"is the model of record", flush=True)
                except Exception as e:
                    print(f"  DEV EVAL FAILED at step {step}: {type(e).__name__}: {e}. "
                          f"Training continues.", flush=True)
                finally:
                    pass
            if is_ddp:
                import torch.distributed as dist
                t_a = torch.tensor([-1 if anneal_start is None else anneal_start,
                                    1 if stop else 0], dtype=torch.long, device=device)
                dist.broadcast(t_a, 0)
                v = int(t_a[0].item())
                anneal_start = None if v < 0 else v
                stop = bool(int(t_a[1].item()))
            if stop:
                break
        model.train()

    if rank == 0:
        core = model.module if is_ddp else model
        end = step + 1
        state = dict(model=core.state_dict(), opt=opt.state_dict(), step=end, cfg=cfg,
                     stopped_early=stop)
        save_ckpt(state, out / "final.pt")
        save_ckpt(state, ckpt)
        print(f"FINETUNE DONE (end step {end}" + (", EARLY-STOPPED — use best.pt" if stop else "") + ")")
    if is_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
