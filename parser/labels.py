"""Dependency-relation label vocabulary (built from train split only)."""
from __future__ import annotations

import json
from pathlib import Path


class DeprelVocab:
    def __init__(self, rels):
        self.rels = rels
        self.idx = {r: i for i, r in enumerate(rels)}

    @classmethod
    def build(cls, sents):
        rels = sorted({t.deprel for s in sents for t in s.tokens if t.deprel and t.deprel != "_"})
        return cls(rels)

    def id(self, r):
        return self.idx.get(r, 0)

    def label(self, i):
        return self.rels[i] if 0 <= i < len(self.rels) else self.rels[0]

    def save(self, path):
        Path(path).write_text(json.dumps(self.rels, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))
