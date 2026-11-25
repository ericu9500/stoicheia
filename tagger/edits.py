"""Edit scripts (form -> lemma) and the label vocabulary.

Scripts operate on *strings*, not the model's char planes: lowercased NFC with the
form-side folded so that graves become acutes and zero-information marks (macron,
breve, dot-below) are stripped. Accent shifts in inflection (ἀνθρώπου -> ἄνθρωπος)
are then literal prefix/suffix replacements, and applying a script yields the fully
accented lemma directly. Lemma-side strings keep everything except casing, which is
captured in a per-script capitalization bit.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

GRAVE, ACUTE = "̀", "́"
_STRIP_MARKS = {"̄", "̆", "̣"}   # macron, breve, dot-below

XPOS_LEN = 9


def form_key(s: str) -> str:
    """Fold a surface form for script/lexicon keys: lowercase, grave->acute, strip
    macron/breve/underdot. NFC output."""
    s = unicodedata.normalize("NFD", s.lower())
    s = s.replace(GRAVE, ACUTE)
    s = "".join(ch for ch in s if ch not in _STRIP_MARKS)
    return unicodedata.normalize("NFC", s)


def lemma_key(s: str) -> str:
    """Lemma side: lowercase + NFC only (macrons/homonym digits kept verbatim)."""
    return unicodedata.normalize("NFC", s.lower())


# Script = (p_cut, p_add, s_cut, s_add, cap) : lemma = p_add + form[p_cut:len-s_cut] + s_add
Script = tuple[int, str, int, str, bool]


def _longest_common_substring(a: str, b: str) -> tuple[int, int, int]:
    """(start_a, start_b, length) of the longest common substring; ties -> smallest
    start_a (align prefixes, since Greek inflection is mostly suffixal)."""
    best = (0, 0, 0)
    m = len(b)
    prev = [0] * (m + 1)
    for i, ca in enumerate(a):
        cur = [0] * (m + 1)
        for j, cb in enumerate(b):
            if ca == cb:
                cur[j + 1] = prev[j] + 1
                l = cur[j + 1]
                if l > best[2]:
                    best = (i - l + 1, j - l + 1, l)
        prev = cur
    return best


def compute_script(form: str, lemma: str) -> Script:
    """form: already form_key()-folded, lowercase. lemma: original casing, NFC."""
    cap = bool(lemma[:1]) and lemma[0] != lemma[0].lower()
    lem = lemma_key(lemma)
    ia, ib, l = _longest_common_substring(form, lem)
    if l == 0:
        return (len(form), lem, 0, "", cap)
    return (ia, lem[:ib], len(form) - ia - l, lem[ib + l:], cap)


def apply_script(form: str, sc: Script) -> str | None:
    p_cut, p_add, s_cut, s_add, _cap = sc
    if len(form) < p_cut + s_cut:
        return None
    return p_add + form[p_cut:len(form) - s_cut or None] + s_add


def script_str(sc: Script) -> str:
    return json.dumps(list(sc), ensure_ascii=False)


@dataclass
class LabelVocab:
    scripts: list[Script] = field(default_factory=list)
    xpos_alpha: list[list[str]] = field(default_factory=list)   # 9 per-position alphabets
    tags: list[str] = field(default_factory=list)               # attested full XPOS tags
    upos: list[str] = field(default_factory=list)
    # lexicon: form_key -> {"lemma\txpos": count} and form_key -> {lemma: count}
    lex_ft: dict = field(default_factory=dict)
    lex_f: dict = field(default_factory=dict)
    # rule table for unencodable (non-Greek) forms: raw form -> (lemma, upos, xpos)
    nongreek: dict = field(default_factory=dict)
    fallback_xpos: str = "u--------"
    fallback_upos: str = "u"

    def __post_init__(self):
        self._sid = {s: i for i, s in enumerate(self.scripts)}
        self._tagset = set(self.tags)
        self._tid = {t: i for i, t in enumerate(self.tags)}
        self._xid = [{c: i for i, c in enumerate(a)} for a in self.xpos_alpha]
        self._uid = {u: i for i, u in enumerate(self.upos)}

    # ---- ids (return -100 for unseen: ignored in the loss, counted in coverage)
    def script_id(self, sc: Script) -> int:
        return self._sid.get(sc, -100)

    def xpos_ids(self, tag: str) -> list[int]:
        tag = (tag or "-" * XPOS_LEN)[:XPOS_LEN].ljust(XPOS_LEN, "-")
        return [self._xid[p].get(c, -100) for p, c in enumerate(tag)]

    def upos_id(self, u: str) -> int:
        return self._uid.get(u, -100)

    def tag_id(self, tag: str) -> int:
        tag = (tag or "-" * XPOS_LEN)[:XPOS_LEN].ljust(XPOS_LEN, "-")
        return self._tid.get(tag, -100)

    @property
    def n_scripts(self):
        return len(self.scripts)

    # ---- build / io
    @classmethod
    def build(cls, sentences, encodable_fn) -> "LabelVocab":
        """sentences: iterable of conllu.Sentence. encodable_fn(form)->bool decides which
        tokens go through the neural path vs the non-Greek rule table."""
        scripts = Counter()
        pos_alpha = [set("-") for _ in range(XPOS_LEN)]
        tags = Counter()
        upos = Counter()
        lex_ft, lex_f = {}, {}
        nongreek = {}
        for sent in sentences:
            for t in sent.tokens:
                tag = (t.xpos or "-" * XPOS_LEN)[:XPOS_LEN].ljust(XPOS_LEN, "-")
                if not encodable_fn(t.form):
                    nongreek.setdefault(t.form, Counter())[(t.lemma, t.upos, tag)] += 1
                    continue
                key = form_key(t.form)
                sc = compute_script(key, t.lemma)
                scripts[sc] += 1
                for p, c in enumerate(tag):
                    pos_alpha[p].add(c)
                tags[tag] += 1
                upos[t.upos] += 1
                lex_ft.setdefault(key + "\t" + tag, Counter())[t.lemma] += 1
                lex_f.setdefault(key, Counter())[t.lemma] += 1
        return cls(
            scripts=[s for s, _ in scripts.most_common()],
            xpos_alpha=[sorted(a) for a in pos_alpha],
            tags=sorted(tags),
            upos=sorted(upos),
            lex_ft={k: dict(c) for k, c in lex_ft.items()},
            lex_f={k: dict(c) for k, c in lex_f.items()},
            nongreek={f: c.most_common(1)[0][0] for f, c in nongreek.items()},
        )

    def save(self, path):
        d = dict(scripts=[list(s) for s in self.scripts], xpos_alpha=self.xpos_alpha,
                 tags=self.tags, upos=self.upos, lex_ft=self.lex_ft, lex_f=self.lex_f,
                 nongreek=self.nongreek, fallback_xpos=self.fallback_xpos,
                 fallback_upos=self.fallback_upos)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "LabelVocab":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        d["scripts"] = [tuple(s[:4]) + (bool(s[4]),) for s in d["scripts"]]
        d["nongreek"] = {k: tuple(v) for k, v in d["nongreek"].items()}
        return cls(**d)
