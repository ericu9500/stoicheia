"""Chu-Liu-Edmonds maximum-spanning-arborescence decoding for non-projective dependency trees.

The biaffine head's greedy per-token argmax can produce non-trees (cycles, multiple roots).
CLE returns the maximum-scoring valid tree rooted at the pseudo-root, which typically lifts
LAS by a few tenths of a point over greedy on non-projective treebanks like AGDT.
"""
from __future__ import annotations

import numpy as np


def chu_liu_edmonds(score):
    """score: (n, n) float, score[h, d] = weight of arc h->d; node 0 is the root (no incoming
    arc). Returns heads: int array length n with heads[d] the selected head of d, heads[0] = -1.
    Maximization. O(n^3) with cycle contraction — fine for sentence-length n."""
    n = score.shape[0]
    S = score.astype(np.float64).copy()
    np.fill_diagonal(S, -np.inf)
    S[:, 0] = -np.inf                      # root has no incoming arc

    # greedily pick the best head for each non-root node
    heads = np.full(n, -1, dtype=np.int64)
    for d in range(1, n):
        heads[d] = int(np.argmax(S[:, d]))

    cyc = _find_cycle(heads, n)
    if cyc is None:
        return heads

    # contract the cycle into a single node and recurse
    cyc_set = set(cyc)
    others = [v for v in range(n) if v not in cyc_set]
    cnode = len(others)                    # index of the contracted node in the subproblem
    remap = {v: i for i, v in enumerate(others)}
    m = len(others) + 1
    S2 = np.full((m, m), -np.inf)
    # cycle-internal arc weight into each cycle node (for the "swap" bookkeeping)
    in_w = {v: S[heads[v], v] for v in cyc}

    # arcs among non-cycle nodes: copy through
    for h in others:
        for d in others:
            if h != d:
                S2[remap[h], remap[d]] = S[h, d]
    # arcs entering the cycle: best over cycle targets, discounted by the broken internal arc
    back = {}                              # (real_head) -> (real cycle node it best enters)
    for h in others:
        best, bestv = -np.inf, None
        for v in cyc:
            w = S[h, v] - in_w[v]
            if w > best:
                best, bestv = w, v
        S2[remap[h], cnode] = best
        back[h] = bestv
    # arcs leaving the cycle: best over cycle sources (root can't be a dependent)
    fwd = {}
    for d in others:
        if d == 0:
            continue
        best, bestv = -np.inf, None
        for v in cyc:
            if S[v, d] > best:
                best, bestv = S[v, d], v
        S2[cnode, remap[d]] = best
        fwd[d] = bestv

    sub = chu_liu_edmonds(S2)

    # expand the contracted solution back to the original nodes
    res = np.full(n, -1, dtype=np.int64)
    for d in others:
        if d == 0:
            continue                       # root keeps head -1
        h_sub = sub[remap[d]]
        if h_sub == cnode:
            res[d] = fwd[d]                # arc came out of the cycle
        else:
            res[d] = others[h_sub]
    # the contracted node's incoming arc: keep the whole cycle except the one broken edge
    h_sub = sub[cnode]
    real_h = others[h_sub]
    enter = back[real_h]                   # cycle node that the external arc enters
    res[enter] = real_h
    for v in cyc:
        if v != enter:
            res[v] = heads[v]              # keep original internal cycle arc
    return res


def _find_cycle(heads, n):
    """Return one cycle (list of nodes) in the head-pointer graph, or None if it's a forest."""
    state = [0] * n                        # 0=unvisited,1=on-stack,2=done
    for start in range(1, n):
        if state[start] != 0:
            continue
        path, v = [], start
        while v > 0 and state[v] == 0:
            state[v] = 1; path.append(v); v = heads[v]
        if v > 0 and state[v] == 1:        # hit a node on the current stack -> cycle
            cyc = path[path.index(v):]
            for u in path:
                state[u] = 2
            return cyc
        for u in path:
            state[u] = 2
    return None


def decode_mst(arc_scores, mask):
    """arc_scores: (B, W, W+1) [col 0 = root]. mask: (B, W) bool. Returns heads (B, W) long
    tensor on cpu with values in [0..W] (0 = root), padded positions = 0 — matching the greedy
    decode's output convention so downstream LAS counting is identical."""
    import torch
    B, W, _ = arc_scores.shape
    out = torch.zeros(B, W, dtype=torch.long)
    sc = arc_scores.detach().float().cpu().numpy()
    lengths = mask.sum(1).cpu().numpy()
    for b in range(B):
        L = int(lengths[b])
        if L == 0:
            continue
        # build (L+1)x(L+1): node 0 = root, nodes 1..L = words; score[h, d]
        S = np.full((L + 1, L + 1), -1e9)
        block = sc[b, :L, :L + 1]                  # (L, L+1): [dep, head_col]
        S[:, 1:] = block.T                          # score[h, d] = arc_scores[dep=d-1, head=h]
        heads = chu_liu_edmonds(S)                  # heads[d] in [0..L]
        out[b, :L] = torch.from_numpy(heads[1:])    # word d (1..L) -> column index (0=root)
    return out


def mst_heads_labels(arc_scores, rel_scores, mask):
    """MST-decoded heads + the label argmax at each chosen head. Mirrors BiaffineHead.decode's
    (heads_cpu, labels_cpu) return so LAS/UAS counting is unchanged."""
    import torch
    pred_heads = decode_mst(arc_scores, mask)          # (B,W) cpu, 0=root
    B, W = pred_heads.shape
    dev = rel_scores.device
    bi = torch.arange(B, device=dev)[:, None].expand(B, W)
    wi = torch.arange(W, device=dev)[None, :].expand(B, W)
    pred_labels = rel_scores[bi, wi, pred_heads.to(dev)].argmax(-1).cpu()
    return pred_heads, pred_labels
