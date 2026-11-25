"""Meter-constrained decoding of the scan head (gated Viterbi).

Vendored from an existing macronization project (scripts/scan_constraints.py grammar
NFAs + scripts/viterbi_meter.py), adapted to per-LETTER probabilities. The scan head
gives P(O/H/L/V) per letter; Viterbi finds the most probable labelling whose non-O
sequence is a valid path through the meter's automaton (last syllable emits V,
anceps/brevis in longo). The gate rejects the constrained path when it costs more
than `theta` log-prob per letter vs raw argmax (protects against grammar mismatch).
"""
from __future__ import annotations

from collections import defaultdict
from math import inf

# ---------------------------------------------------------------- grammars
PRINCEPS = (("H",),)
BICEPS = (("H",), ("L", "L"))
BREVE = (("L",),)
FINAL = (("H",), ("L",))
ANCEPS = (("H",), ("L",), ("L", "L"))
LONGUM = (("H",), ("L", "L"))

HEX = [PRINCEPS, BICEPS] * 5 + [PRINCEPS, FINAL]
PENTA = ([PRINCEPS, BICEPS] * 2 + [PRINCEPS]
         + [PRINCEPS, BREVE, BREVE, PRINCEPS, BREVE, BREVE, FINAL])
IA6 = ([ANCEPS, LONGUM, BREVE, LONGUM] * 2 + [ANCEPS, LONGUM, BREVE, FINAL])
TROCH_TET = [LONGUM, BREVE, LONGUM, ANCEPS] * 3 + [LONGUM, BREVE, FINAL]
ANAP_FOOT = (("L", "L", "H"), ("H", "H"), ("H", "L", "L"))
ANAP_TET = [ANAP_FOOT] * 7 + [FINAL]
CHOL = ([ANCEPS, LONGUM, BREVE, LONGUM] * 2 + [ANCEPS, LONGUM, (("H",),), FINAL])
BREVE_C = (("L",), ("L", "L"))
IA6C = ([ANCEPS, LONGUM, BREVE_C, LONGUM] * 2 + [ANCEPS, LONGUM, BREVE_C, FINAL])
GRAMMARS = {"hex": HEX, "penta": PENTA, "ia6": IA6,
            "troch_tet": TROCH_TET, "anap_tet": ANAP_TET, "chol": CHOL, "ia6c": IA6C}

# corpus meter names -> grammar keys (everything else: no stichic grammar)
METER_MAP = {
    "dactylic hexameter": "hex", "hexameter": "hex",
    "dactylic pentameter": "penta", "pentameter": "penta",
    "iambic trimeter (tragic)": "ia6",
    "iambic trimeter (anapaestic)": "ia6c",
    "ia6": "ia6", "ia6g": "ia6",
    "tr7": "troch_tet", "!_tr7": "troch_tet",
    "an4": "anap_tet", "an4cat": "anap_tet",
    "choliambic": "chol",
}


def build_nfa(spec):
    trans = defaultdict(set)
    for p, pos in enumerate(spec):
        for ri, real in enumerate(pos):
            prev = ("B", p)
            for j, w in enumerate(real):
                nxt = ("B", p + 1) if j == len(real) - 1 else ("I", p, ri, j + 1)
                trans[(prev, w)].add(nxt)
                prev = nxt
    return dict(trans), ("B", 0), ("B", len(spec))


NFAS = {name: build_nfa(spec) for name, spec in GRAMMARS.items()}

O, H, L, V = 0, 1, 2, 3


def viterbi_decode(logP, meter):
    """logP: (n,4) array-like of log-probs per letter -> (labels|None, score)."""
    trans, start, accept = NFAS[meter]
    N = len(logP)
    dp = [dict() for _ in range(N + 1)]
    dp[0][start] = (0.0, None)
    for i in range(N):
        if not dp[i]:
            break
        lpO, lpH, lpL, lpV = logP[i]
        for s, (sc, _) in dp[i].items():
            cand = sc + lpO                      # emit O: stay
            if cand > dp[i + 1].get(s, (-inf,))[0]:
                dp[i + 1][s] = (cand, (s, O))
            for w, lpw, wsym in ((H, lpH, "H"), (L, lpL, "L")):
                for s2 in trans.get((s, wsym), ()):
                    lab = V if s2 == accept else w
                    cand = sc + (lpV if s2 == accept else lpw)
                    if cand > dp[i + 1].get(s2, (-inf,))[0]:
                        dp[i + 1][s2] = (cand, (s, lab))
    if accept not in dp[N]:
        return None, -inf
    labels = [O] * N
    s = accept
    score = dp[N][s][0]
    for i in range(N, 0, -1):
        prev, lab = dp[i][s][1]
        labels[i - 1] = lab
        s = prev
    return labels, score


def best_meter(logP, meters=("hex", "penta", "ia6", "ia6c", "troch_tet",
                             "anap_tet", "chol")):
    """(meter, labels, per-letter score) of the best-scoring valid grammar."""
    best = (None, None, -inf)
    for mt in meters:
        labs, sc = viterbi_decode(logP, mt)
        if labs is not None and sc / len(logP) > best[2]:
            best = (mt, labs, sc / len(logP))
    return best


def gated_decode(logP, meter, theta=0.1):
    """Viterbi unless it costs > theta log-prob/letter vs argmax.
    -> (labels, applied: bool)."""
    raw = [max(p) for p in logP]
    labs, sc = viterbi_decode(logP, meter)
    if labs is not None and (sum(raw) - sc) / max(1, len(logP)) <= theta:
        return labs, True
    return [int(max(range(4), key=lambda k: p[k])) for p in logP], False


def gated_auto(logP, theta=0.1):
    """Auto-detect meter, then gate. -> (meter|None, labels, applied)."""
    mt, labs, _ = best_meter(logP)
    if mt is not None:
        raw = [max(p) for p in logP]
        _, sc = viterbi_decode(logP, mt)
        if (sum(raw) - sc) / max(1, len(logP)) <= theta:
            return mt, labs, True
    return mt, [int(max(range(4), key=lambda k: p[k])) for p in logP], False
