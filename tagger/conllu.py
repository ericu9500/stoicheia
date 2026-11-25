"""Minimal stdlib CoNLL-U reader/writer for the OGA kfold treebank.

Keeps only what the tagger needs (FORM/LEMMA/UPOS/XPOS per syntactic word) but
round-trips every other column and comment untouched, so predictions can be written
into a copy of the gold file and scored with the official conll18 script.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Token:
    tid: str          # "1", "2", ... (kept as string; MWT ranges and empty nodes never land here)
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str


@dataclass
class Sentence:
    tokens: list[Token] = field(default_factory=list)
    # raw lines in original order, as (kind, payload): kind "comment" | "mwt" | "empty" -> raw
    # line, kind "token" -> index into tokens. Preserves exact file structure on write.
    lines: list[tuple[str, object]] = field(default_factory=list)


def read_conllu(path) -> Iterator[Sentence]:
    sent = Sentence()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if sent.lines:
                    yield sent
                    sent = Sentence()
                continue
            if line.startswith("#"):
                sent.lines.append(("comment", line))
                continue
            cols = line.split("\t")
            tid = cols[0]
            if "-" in tid:
                sent.lines.append(("mwt", line))
            elif "." in tid:
                sent.lines.append(("empty", line))
            else:
                sent.lines.append(("token", len(sent.tokens)))
                sent.tokens.append(Token(*cols[:10]))
    if sent.lines:
        yield sent


def write_conllu(sents, preds, path):
    """preds: list (per sentence) of lists of (lemma, upos, xpos) aligned with sent.tokens."""
    with open(path, "w", encoding="utf-8") as f:
        for sent, ps in zip(sents, preds):
            for kind, payload in sent.lines:
                if kind == "token":
                    t = sent.tokens[payload]
                    lemma, upos, xpos = ps[payload]
                    f.write("\t".join([t.tid, t.form, lemma, upos, xpos,
                                       t.feats, t.head, t.deprel, t.deps, t.misc]) + "\n")
                else:
                    f.write(payload + "\n")
            f.write("\n")
