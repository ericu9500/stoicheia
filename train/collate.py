"""Collate shard records into model-ready batches: noise, then bin-pack.

Pipeline per record: raw planes (chars/boundary/dia/cap/punct) -> noise_sequence (elastic
patterns change length) -> pack B records into fixed-width rows, building seg_id for
doc-masked attention (multiple documents per row, ~99% fill vs ~20% one-per-row).

Aux-head labels (boundary/dia/cap/punct) are supervised only where the character was masked
OR the corresponding input channel was dropped for that position — elsewhere it would just
be copy-through, teaching the model nothing.
"""
from __future__ import annotations

import numpy as np
import torch

from train.noising import NoiseConfig, noise_sequence

UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6
# Metadata-conditioning ids (insc/data/meta_vocab.py's UNK_REGION/UNK_CENTURY) -- kept as
# plain constants here, not imported, so base pretraining (which never sets these) doesn't
# pick up a dependency on the insc-only package. Records without region_id/century_id
# (every non-insc shard) fall back to these UNK rows; CharBertEncoder only reads them at
# all when its optional e_region/e_century tables are enabled (n_region/n_century > 0).
UNK_REGION, UNK_CENTURY = 14, 15


def _dropped_region_century(rec, cfg, g):
    """Per-DOCUMENT metadata dropout (not per-position): with probability p_region_none /
    p_century_none, force UNK regardless of whether the true value is known -- so the model
    is trained under both "metadata given" and "metadata withheld" conditions (a real
    fragment's provenance/date is often genuinely unknown at inference too). Independent
    draws: knowing a find-spot doesn't imply knowing the date or vice versa."""
    region_id = rec.get("region_id", UNK_REGION)
    century_id = rec.get("century_id", UNK_CENTURY)
    if cfg.p_region_none > 0 and torch.rand(1, generator=g).item() < cfg.p_region_none:
        region_id = UNK_REGION
    if cfg.p_century_none > 0 and torch.rand(1, generator=g).item() < cfg.p_century_none:
        century_id = UNK_CENTURY
    return region_id, century_id


def pack_batch(record_iter, cfg: NoiseConfig, T_char, rows, g):
    """Greedy sequence packing: pull records, noise them, bin-pack into `rows` rows of width
    T_char. Each doc within a row gets a distinct seg id (1,2,..) so attention never crosses
    a document boundary (block-diagonal, not a context discount)."""
    B = rows
    ids = torch.full((B, T_char), cfg.pad_id, dtype=torch.long)
    bnd_in = torch.full((B, T_char), UNK_BND, dtype=torch.long)
    dia_in = torch.full((B, T_char), UNK_DIA, dtype=torch.long)
    punct_in = torch.full((B, T_char), UNK_PUNCT, dtype=torch.long)
    region_in = torch.full((B, T_char), UNK_REGION, dtype=torch.long)
    century_in = torch.full((B, T_char), UNK_CENTURY, dtype=torch.long)
    seg = torch.zeros(B, T_char, dtype=torch.long)
    labels = torch.full((B, T_char), -100, dtype=torch.long)
    loss_w = torch.zeros(B, T_char)
    bnd_lab = torch.full((B, T_char), -100, dtype=torch.long)
    dia_lab = torch.full((B, T_char), -100, dtype=torch.long)
    cap_lab = torch.full((B, T_char), -100, dtype=torch.long)
    punct_lab = torch.full((B, T_char), -100, dtype=torch.long)

    for b in range(B):
        cpos = 0          # char cursor in this row
        doc = 0            # doc id within row
        misses = 0         # consecutive records that didn't fit the tail
        while True:
            rec = next(record_iter)
            chars = torch.from_numpy(rec["chars"].astype(np.int64))
            boundary = torch.from_numpy(rec["boundary"].astype(np.int64))
            real_lac = (torch.from_numpy(rec["is_real_lacuna"]) if "is_real_lacuna" in rec
                       else None)
            out = noise_sequence(chars, boundary.to(torch.uint8), cfg, g, is_real_lacuna=real_lac)
            L = out["input_ids"].numel()
            if L > T_char:          # over-long single doc: truncate to fit an empty row
                L = T_char
            if cpos + L > T_char:
                if cpos == 0:       # row empty: force-place a truncated copy so we never stall
                    L = T_char
                else:               # try a few more (smaller) records before giving up on this row
                    misses += 1
                    if misses >= 6:
                        break
                    continue
            misses = 0
            doc += 1
            sl = slice(cpos, cpos + L)
            ids[b, sl] = out["input_ids"][:L]
            labels[b, sl] = out["labels"][:L]
            loss_w[b, sl] = out["loss_w"][:L]
            seg[b, sl] = doc
            region_id, century_id = _dropped_region_century(rec, cfg, g)
            region_in[b, sl] = region_id
            century_in[b, sl] = century_id
            kb = out["keep_bnd_mask"][:L]; kd = out["keep_dia_mask"][:L]; kp = out["keep_punct_mask"][:L]
            bnd_true = out["boundary"][:L].long().clamp(max=2)
            bnd_in[b, sl] = torch.where(kb, bnd_true, torch.full_like(bnd_true, UNK_BND))
            if not out["rebuilt"]:
                dia_true = torch.from_numpy(rec["dia"].astype(np.int64))[:L]
                dia_in[b, sl] = torch.where(kd, dia_true, torch.full_like(dia_true, UNK_DIA))
                punct_true = torch.from_numpy(rec["punct"].astype(np.int64))[:L]
                punct_in[b, sl] = torch.where(kp, punct_true, torch.full_like(punct_true, UNK_PUNCT))
            masked = ids[b, sl] == cfg.mask_id
            if not out["rebuilt"]:
                Lc = min(L, chars.numel())
                # real-lacuna positions have no ground truth for ANY channel (not just chars)
                # -- exclude them from aux supervision too, same as the char loss.
                not_lac = (~real_lac[:Lc] if real_lac is not None
                          else torch.ones(Lc, dtype=torch.bool))
                sup_b = (masked[:Lc] | (~kb[:Lc])) & not_lac
                sup_d = (masked[:Lc] | (~kd[:Lc])) & not_lac
                sup_p = (masked[:Lc] | (~kp[:Lc])) & not_lac
                sup_c = masked[:Lc] & not_lac
                bnd_t = boundary[:Lc]; dia_t = torch.from_numpy(rec["dia"].astype(np.int64))[:Lc]
                cap_t = torch.from_numpy(rec["cap"].astype(np.int64))[:Lc]
                punct_t = torch.from_numpy(rec["punct"].astype(np.int64))[:Lc]
                bnd_lab[b, cpos:cpos+Lc] = torch.where(sup_b, bnd_t, torch.full_like(bnd_t, -100))
                dia_lab[b, cpos:cpos+Lc] = torch.where(sup_d, dia_t, torch.full_like(dia_t, -100))
                cap_lab[b, cpos:cpos+Lc] = torch.where(sup_c, cap_t, torch.full_like(cap_t, -100))
                punct_lab[b, cpos:cpos+Lc] = torch.where(sup_p, punct_t, torch.full_like(punct_t, -100))
            cpos += L
            if cpos >= T_char - 8:
                break
    return dict(
        input_ids=ids, boundary=bnd_in, dia=dia_in, punct=punct_in, seg_id=seg,
        region=region_in, century=century_in,
        labels=labels, loss_w=loss_w, bnd_lab=bnd_lab, dia_lab=dia_lab, cap_lab=cap_lab,
        punct_lab=punct_lab,
    )


def collate(records, cfg: NoiseConfig, T_char, g):
    """records: list of dicts with chars/boundary/dia/cap/punct (np.uint8 arrays). One doc
    per row (no packing) — used for eval batches where records are already pre-selected."""
    B = len(records)
    ids = torch.full((B, T_char), cfg.pad_id, dtype=torch.long)
    bnd_in = torch.full((B, T_char), UNK_BND, dtype=torch.long)
    dia_in = torch.full((B, T_char), UNK_DIA, dtype=torch.long)
    punct_in = torch.full((B, T_char), UNK_PUNCT, dtype=torch.long)
    region_in = torch.full((B, T_char), UNK_REGION, dtype=torch.long)
    century_in = torch.full((B, T_char), UNK_CENTURY, dtype=torch.long)
    seg = torch.zeros(B, T_char, dtype=torch.long)
    labels = torch.full((B, T_char), -100, dtype=torch.long)
    loss_w = torch.zeros(B, T_char)
    bnd_lab = torch.full((B, T_char), -100, dtype=torch.long)
    dia_lab = torch.full((B, T_char), -100, dtype=torch.long)
    cap_lab = torch.full((B, T_char), -100, dtype=torch.long)
    punct_lab = torch.full((B, T_char), -100, dtype=torch.long)

    for b, rec in enumerate(records):
        chars = torch.from_numpy(rec["chars"].astype(np.int64))
        boundary = torch.from_numpy(rec["boundary"].astype(np.int64))
        real_lac = (torch.from_numpy(rec["is_real_lacuna"]) if "is_real_lacuna" in rec
                   else None)
        out = noise_sequence(chars, boundary.to(torch.uint8), cfg, g, is_real_lacuna=real_lac)
        seqlen = min(out["input_ids"].numel(), T_char)
        ids[b, :seqlen] = out["input_ids"][:seqlen]
        labels[b, :seqlen] = out["labels"][:seqlen]
        loss_w[b, :seqlen] = out["loss_w"][:seqlen]
        seg[b, :seqlen] = b + 1
        region_id, century_id = _dropped_region_century(rec, cfg, g)
        region_in[b, :seqlen] = region_id
        century_in[b, :seqlen] = century_id
        kb = out["keep_bnd_mask"][:seqlen]; kd = out["keep_dia_mask"][:seqlen]
        kp = out["keep_punct_mask"][:seqlen]
        bnd_true = out["boundary"][:seqlen].long().clamp(max=2)
        bnd_in[b, :seqlen] = torch.where(kb, bnd_true, torch.full_like(bnd_true, UNK_BND))
        if not out["rebuilt"]:
            dia_true = torch.from_numpy(rec["dia"].astype(np.int64))[:seqlen]
            dia_in[b, :seqlen] = torch.where(kd, dia_true, torch.full_like(dia_true, UNK_DIA))
            punct_true = torch.from_numpy(rec["punct"].astype(np.int64))[:seqlen]
            punct_in[b, :seqlen] = torch.where(kp, punct_true, torch.full_like(punct_true, UNK_PUNCT))

        masked = ids[b] == cfg.mask_id
        if not out["rebuilt"]:
            L = min(seqlen, chars.numel())
            not_lac = (~real_lac[:L] if real_lac is not None
                      else torch.ones(L, dtype=torch.bool))
            sup_b = (masked[:L] | (~kb[:L])) & not_lac
            sup_d = (masked[:L] | (~kd[:L])) & not_lac
            sup_p = (masked[:L] | (~kp[:L])) & not_lac
            sup_c = masked[:L] & not_lac
            cap_t = torch.from_numpy(rec["cap"].astype(np.int64))
            bnd_t = boundary.clone()
            dia_t = torch.from_numpy(rec["dia"].astype(np.int64))
            punct_t = torch.from_numpy(rec["punct"].astype(np.int64))
            bnd_lab[b, :L] = torch.where(sup_b, bnd_t[:L], torch.full_like(bnd_t[:L], -100))
            dia_lab[b, :L] = torch.where(sup_d, dia_t[:L], torch.full_like(dia_t[:L], -100))
            cap_lab[b, :L] = torch.where(sup_c, cap_t[:L], torch.full_like(cap_t[:L], -100))
            punct_lab[b, :L] = torch.where(sup_p, punct_t[:L], torch.full_like(punct_t[:L], -100))

    return dict(
        input_ids=ids, boundary=bnd_in, dia=dia_in, punct=punct_in, seg_id=seg,
        region=region_in, century=century_in,
        labels=labels, loss_w=loss_w, bnd_lab=bnd_lab, dia_lab=dia_lab, cap_lab=cap_lab,
        punct_lab=punct_lab,
    )
