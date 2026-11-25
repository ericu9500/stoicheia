"""Intrinsic eval — the metrics tracked during pretraining, on a genuinely held-out split
(every HOLDOUT_MOD-th record, never trained on; see train/data.py::eligible).

  bits_per_char   masked-char cross-entropy in bits, at a fixed reference mask rate (t=0.30)
  boundary_f1     word-boundary prediction F1 with the boundary channel forced UNKNOWN
                  (the model must segment scriptio continua from scratch)
  dia_acc         diacritic restoration accuracy, channel forced UNKNOWN
  punct_acc       punctuation restoration accuracy, channel forced UNKNOWN
  demo            a fixed smoke reconstruction, for a human sanity check every checkpoint
"""
from __future__ import annotations

import json, math
from pathlib import Path

import numpy as np
import torch

from data.normalize import ALPHABET
from model.char_bert import CharBertConfig, CharBertEncoder
from train.collate import collate
from train.noising import NoiseConfig

ALIST = list(ALPHABET)
UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6


def held_out_records(shards, n, seed=1234):
    import pyarrow.parquet as pq
    d = Path(shards)
    idx = pq.read_table(d / "index.parquet")
    offs = idx.column("offset").to_numpy(); lens = idx.column("length").to_numpy()
    tier = idx.column("tier").to_numpy(zero_copy_only=False)
    chars = np.memmap(d / "chars.bin", dtype=np.uint8, mode="r")
    bnd = np.memmap(d / "boundary.bin", dtype=np.uint8, mode="r")
    dia = np.memmap(d / "dia.bin", dtype=np.uint8, mode="r")
    cap = np.memmap(d / "cap.bin", dtype=np.uint8, mode="r")
    punct = np.memmap(d / "punct.bin", dtype=np.uint8, mode="r")
    from train.data import HOLDOUT_MOD
    gold = np.flatnonzero((tier == "pristine") & (lens >= 64) & (lens <= 4096))
    gold = gold[gold % HOLDOUT_MOD == 0]
    rng = np.random.default_rng(seed)
    pick = rng.choice(gold, size=min(n, len(gold)), replace=False)
    recs = []
    for i in pick:
        o, l = int(offs[i]), int(lens[i])
        recs.append(dict(chars=np.array(chars[o:o+l]), boundary=np.array(bnd[o:o+l]),
                         dia=np.array(dia[o:o+l]), cap=np.array(cap[o:o+l]),
                         punct=np.array(punct[o:o+l])))
    return recs


@torch.no_grad()
def evaluate(model, recs, device, T=1024, micro=8, rate=0.30):
    model.eval()
    # fixed reference noise: iid at t=rate, boundary/dia/punct channels forced UNKNOWN
    ncfg = NoiseConfig(w_span=0, w_word=0, w_elastic=0, w_iid=1, w_halfword=0, w_substitute=0,
                       beta_a=1e6, beta_b=1e6 * (1 - rate) / rate,
                       p_bnd_full=0.0, p_bnd_none=1.0, p_dia_full=0.0, p_dia_none=1.0,
                       p_punct_full=0.0, p_punct_none=1.0)
    g = torch.Generator().manual_seed(0)
    tot_bits = tot_n = 0
    b_tp = b_fp = b_fn = 0
    dia_ok = dia_n = 0
    punct_ok = punct_n = 0
    for i in range(0, len(recs), micro):
        batch = collate(recs[i:i+micro], ncfg, T, g)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch)
        lab = batch["labels"]; m = lab != -100
        if m.any():
            logp = torch.log_softmax(out["char"][m].float(), -1)
            nll = -logp.gather(1, lab[m][:, None]).squeeze(1)
            tot_bits += (nll.sum() / math.log(2)).item(); tot_n += int(m.sum())
        bl = batch["bnd_lab"]; bm = bl != -100
        if bm.any():
            pred = out["boundary"][bm].argmax(-1)
            gold_fin = bl[bm] >= 1; pred_fin = pred >= 1
            b_tp += int((gold_fin & pred_fin).sum()); b_fp += int((~gold_fin & pred_fin).sum())
            b_fn += int((gold_fin & ~pred_fin).sum())
        dl = batch["dia_lab"]; dm = dl != -100
        if dm.any():
            dia_ok += int((out["dia"][dm].argmax(-1) == dl[dm]).sum()); dia_n += int(dm.sum())
        pl = batch["punct_lab"]; pm = pl != -100
        if pm.any():
            punct_ok += int((out["punct"][pm].argmax(-1) == pl[pm]).sum()); punct_n += int(pm.sum())
    prec = b_tp / (b_tp + b_fp + 1e-9); rec = b_tp / (b_tp + b_fn + 1e-9)
    bf1 = 2 * prec * rec / (prec + rec + 1e-9)
    model.train()
    return dict(bits_per_char=round(tot_bits / max(tot_n, 1), 4),
                boundary_f1=round(bf1, 4), dia_acc=round(dia_ok / max(dia_n, 1), 4),
                punct_acc=round(punct_ok / max(punct_n, 1), 4), n_eval=tot_n)


@torch.no_grad()
def restore_demo(model, device, text="καιολογοσ", mask_slice=(2, 5)):
    """Mask chars [i:j] and show the model's top-1 reconstruction (the κα---ογοσ demo)."""
    from data.normalize import Stats, normalize_record
    st = Stats()
    r = normalize_record(text, st, with_punct=True)
    if r is None:
        return "?"
    chars, boundary, dia, cap, punct = r
    ids = torch.tensor(chars, dtype=torch.long)
    inp = ids.clone(); inp[mask_slice[0]:mask_slice[1]] = 24  # mask_id
    batch = dict(input_ids=inp[None].to(device),
                 boundary=torch.full((1, len(ids)), UNK_BND, device=device),
                 dia=torch.full((1, len(ids)), UNK_DIA, device=device),
                 punct=torch.full((1, len(ids)), UNK_PUNCT, device=device),
                 seg_id=torch.ones(1, len(ids), dtype=torch.long, device=device))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch)
    pred = out["char"][0].argmax(-1).cpu()
    recon = "".join(ALIST[c] if c < 24 else "?" for c in pred[mask_slice[0]:mask_slice[1]])
    full = "".join(ALIST[inp[k]] if inp[k] < 24 else recon[k-mask_slice[0]]
                   for k in range(len(ids)))
    return full


def load_model(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    c = sd["cfg"]
    # metadata-conditioned finetunes (finetune_whole.py) carry e_region/e_century
    # embeddings; size them from the state dict itself so both kinds of checkpoint load
    msd = sd["model"]
    n_region = msd["e_region.weight"].shape[0] if "e_region.weight" in msd else 0
    n_century = msd["e_century.weight"].shape[0] if "e_century.weight" in msd else 0
    mcfg = CharBertConfig(attn_impl="sdpa", d_model=c["d_model"], n_heads=c["d_model"] // 64,
                          depth=c["depth"], char_window=c["char_window"],
                          qk_norm=c.get("qk_norm", True),
                          n_region=n_region, n_century=n_century)
    model = CharBertEncoder(mcfg).to(device)
    model.load_state_dict(msd)
    return model, c
