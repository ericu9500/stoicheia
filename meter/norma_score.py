"""Canonical scorer for the Norma benchmark (macronize + syllabify).

Self-contained on purpose (stdlib only) so this can be lifted out and dropped
straight into the Norma dataset repo for anyone to score their own predictions
against -- it does not import anything else from Stoicheia.

Both tasks are scored by projecting the annotation onto LETTER ORDINALS (the
n-th actual Greek letter in the line, ignoring whitespace/punctuation/marks
entirely) rather than by diffing the annotated strings themselves. This makes
scoring robust to cosmetic differences in mark/bracket placement around
whitespace -- e.g. for syllabify, a predicted "{ word}" scores identical to a
gold " {word}": the leading space carries no ordinal and is not itself part of
either span, so which side of the bracket it sits on cannot change which
letters the label applies to.

Usage:
  python -m meter.norma_score --gold test.jsonl --pred pred.jsonl [--task macronize|syllabify|both]

`gold` is Norma's own format: one {"text", "source", "task"} object per line
(see https://huggingface.co/datasets/Ericu950/norma). `pred` must have the same
number of lines in the same order, each a plain annotated string (macronize:
"_"/"^" after long/short letters; syllabify: "[heavy]"/"{light}" spans) -- or a
{"text": "..."} object with the same content under "text".
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict

MAC_LONG, MAC_SHORT = 0, 1
SCAN_HEAVY, SCAN_LIGHT, SCAN_VERSE = 1, 2, 3

_LONG_MARKS = {"_", "̄"}    # ASCII underscore, combining macron
_SHORT_MARKS = {"^", "̆"}   # ASCII caret, combining breve
_ALL_MARKS = _LONG_MARKS | _SHORT_MARKS

_GREEK_BLOCKS = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))  # Greek+Coptic, Greek Extended


def _is_letter(ch: str) -> bool:
    """Is this (possibly precomposed) character one Greek letter, for ordinal-
    counting purposes? Decomposes first so accented/breathed precomposed forms
    (e.g. "ά", "ᾧ") are recognized via their base letter."""
    base = unicodedata.normalize("NFD", ch)[0]
    cp = ord(base)
    return any(lo <= cp <= hi for lo, hi in _GREEK_BLOCKS) and unicodedata.category(base).startswith("L")


def parse_macron_line(marked: str):
    """"βα^ρύκτυ^πος" -> (plain, {letter_ordinal: MAC_LONG|MAC_SHORT})."""
    nfd = unicodedata.normalize("NFD", marked)
    kept, labels = [], {}
    ordinal = -1
    for ch in nfd:
        if ch in _ALL_MARKS:
            if ordinal >= 0:
                labels[ordinal] = MAC_LONG if ch in _LONG_MARKS else MAC_SHORT
            continue
        if _is_letter(ch):
            ordinal += 1
        kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept)), labels


_SYL = re.compile(r"\[([^\]]*)\]|\{([^}]*)\}")


def parse_scan_line(bracketed: str):
    """"[ὦ] [παῖ] {τέ}[λος]" -> (plain, {letter_ordinal: scan class}), weight on
    the LAST letter of each span; the line's last labeled letter becomes
    SCAN_VERSE (brevis in longo). Whitespace on either side of a bracket is
    inert: it carries no ordinal, so it cannot shift which letter a label
    lands on regardless of which side of the bracket it's written on."""
    plain_parts, labels = [], {}
    ordinal = -1
    pos = 0
    last_labeled = None

    def advance(text):
        nonlocal ordinal
        last = None
        for ch in text:
            if _is_letter(ch):
                ordinal += 1
                last = ordinal
        plain_parts.append(text)
        return last

    for m in _SYL.finditer(bracketed):
        advance(bracketed[pos:m.start()])
        text, weight = ((m.group(1), SCAN_HEAVY) if m.group(1) is not None
                        else (m.group(2), SCAN_LIGHT))
        last = advance(text)
        if last is not None:
            labels[last] = weight
            last_labeled = last
        pos = m.end()
    advance(bracketed[pos:])
    if last_labeled is None:
        return None
    labels[last_labeled] = SCAN_VERSE
    return unicodedata.normalize("NFC", "".join(plain_parts)), labels


def _text_of(row) -> str:
    return row["text"] if isinstance(row, dict) else row


def mac_metrics(pairs):
    """pairs: [(gold, pred)] in {MAC_LONG, MAC_SHORT} -> acc, balanced acc, per-class F1."""
    if not pairs:
        return None
    n = len(pairs)
    acc = sum(g == p for g, p in pairs) / n
    out = {"n": n, "acc": round(acc, 4)}
    recalls = []
    for cls, name in [(MAC_LONG, "long"), (MAC_SHORT, "short")]:
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else None
        out[f"{name}_f1"] = round(2 * prec * rec / (prec + rec)
                                   if rec is not None and prec + rec else 0.0, 4)
        if rec is not None:
            recalls.append(rec)
    out["bal_acc"] = round(sum(recalls) / len(recalls), 4) if recalls else None
    return out


def scan_metrics(pairs):
    """pairs: [(gold 0..3, pred 0..3)] -> acc, balanced acc, boundary-F1, weight acc
    (heavy-vs-light accuracy conditional on both sides agreeing a boundary exists)."""
    if not pairs:
        return None
    n = len(pairs)
    acc = sum(g == p for g, p in pairs) / n
    recalls = []
    for c in range(4):
        tot = sum(1 for g, _ in pairs if g == c)
        if tot:
            recalls.append(sum(1 for g, p in pairs if g == c and p == c) / tot)
    tp = sum(1 for g, p in pairs if g > 0 and p > 0)
    fp = sum(1 for g, p in pairs if g == 0 and p > 0)
    fn = sum(1 for g, p in pairs if g > 0 and p == 0)
    w_pairs = [(g, p) for g, p in pairs if g in (SCAN_HEAVY, SCAN_LIGHT) and p in (SCAN_HEAVY, SCAN_LIGHT)]
    return dict(n=n, acc=round(acc, 4),
                bal_acc=round(sum(recalls) / len(recalls), 4) if recalls else None,
                boundary_f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                weight_acc=round(sum(g == p for g, p in w_pairs) / len(w_pairs), 4)
                if w_pairs else None)


def score_macronize(gold_rows, pred_rows):
    """gold_rows/pred_rows: parallel lists of macronize-annotated strings (or
    {"text": ...} dicts). -> (overall metrics, {source: metrics})."""
    pairs, by_source = [], defaultdict(list)
    for g, p in zip(gold_rows, pred_rows):
        gplain, glabels = parse_macron_line(_text_of(g))
        pplain, plabels = parse_macron_line(_text_of(p))
        assert gplain == pplain, f"plain-text mismatch: {gplain!r} vs {pplain!r}"
        source = g.get("source") if isinstance(g, dict) else None
        for k, gv in glabels.items():
            pv = plabels.get(k, MAC_LONG)  # old convention: default-to-long if unmarked
            pairs.append((gv, pv))
            if source:
                by_source[source].append((gv, pv))
    per_source = {s: mac_metrics(ps) for s, ps in by_source.items()}
    return mac_metrics(pairs), per_source


def score_syllabify(gold_rows, pred_rows):
    """Same shape as score_macronize, but for bracketed syllabify text. Comparison
    happens over the FULL per-letter array (0 = no boundary at that letter), not
    just the labeled ordinals, so a boundary predicted at the wrong letter shows
    up as a mismatch on both the letter that should have it and the one that
    wrongly does."""
    pairs, by_source = [], defaultdict(list)
    for g, p in zip(gold_rows, pred_rows):
        gparsed = parse_scan_line(_text_of(g))
        pparsed = parse_scan_line(_text_of(p))
        if gparsed is None:
            continue
        gplain, glabels = gparsed
        pplain, plabels = pparsed if pparsed is not None else (gplain, {})
        assert gplain == pplain, f"plain-text mismatch: {gplain!r} vs {pplain!r}"
        n = sum(1 for ch in gplain if _is_letter(ch))
        source = g.get("source") if isinstance(g, dict) else None
        for k in range(n):
            gv, pv = glabels.get(k, 0), plabels.get(k, 0)
            pairs.append((gv, pv))
            if source:
                by_source[source].append((gv, pv))
    per_source = {s: scan_metrics(ps) for s, ps in by_source.items()}
    return scan_metrics(pairs), per_source


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="Norma-format jsonl (text/source/task)")
    ap.add_argument("--pred", required=True, help="parallel jsonl, same order, annotated text")
    ap.add_argument("--task", choices=["macronize", "syllabify", "both"], default="both")
    a = ap.parse_args()

    gold_all, pred_all = _load_jsonl(a.gold), _load_jsonl(a.pred)
    assert len(gold_all) == len(pred_all), \
        f"{a.gold}: {len(gold_all)} rows vs {a.pred}: {len(pred_all)} rows"

    for task, scorer in [("macronize", score_macronize), ("syllabify", score_syllabify)]:
        if a.task not in (task, "both"):
            continue
        pairs = [(g, p) for g, p in zip(gold_all, pred_all) if g.get("task") == task]
        if not pairs:
            continue
        gold_rows, pred_rows = zip(*pairs)
        overall, per_source = scorer(list(gold_rows), list(pred_rows))
        print(f"=== {task} ===")
        print(f"  overall: {json.dumps(overall)}")
        for s in sorted(per_source):
            print(f"  {s}: {json.dumps(per_source[s])}")


if __name__ == "__main__":
    main()
