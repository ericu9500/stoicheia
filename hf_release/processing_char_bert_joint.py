"""HF-Hub-compatible processor for Stoicheia-tagger-parser.

Turns one or more already word-tokenized sentences into the model's input tensors (the same
four character planes as the base CharBertProcessor, plus a `word_id` array pooling characters
into words, a `cap` plane -- an actual model input for this fine-tuned checkpoint, unlike the
base model where capitalization is output-only -- and a `seg_id` plane), and decodes the
model's raw logits back into predictions. The model's NATIVE tag/relation inventory is the
Perseus/AGDT scheme (9-position morphological tags; Prague-style dependency relations like
SBJ/OBJ/ATR/COORD/AuxP -- the scheme its OGA training treebank uses), not Universal
Dependencies. Pass `ud=True` to `decode()` for Universal Dependencies-style output instead
(UPOS/FEATS/deprel) -- see `to_ud()` below for the conversion and its scope/limitations.

Interface: call the processor with `List[str]` (one already-tokenized sentence, e.g.
`["ὁ", "ἄνθρωπος", "τρέχει", "."]`) or `List[List[str]]` (a batch of such sentences) --
i.e. tokens are CoNLL-U-style FORMs, punctuation already split into its own element. This
mirrors tagger/dataset.py's `sent.tokens` convention directly (each token's FORM is encoded
independently) without pulling in a real Greek word tokenizer as a dependency; a convenience
`.tokenize(text)` staticmethod is provided for plain-text input (naive whitespace split +
peeling off trailing sentence punctuation), which is enough for a usage-example sentence but
is not a substitute for real tokenization on running text.

Not a `PreTrainedTokenizer` subclass, for the same reason as the base CharBertProcessor: the
underlying encoding is parallel per-letter planes, not a single token-id stream. Follows the
same `register_for_auto_class` mechanism, so `AutoProcessor.from_pretrained(repo_id,
trust_remote_code=True)` works the same way.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch

PAD_ID = 26
UNK_BND, UNK_DIA, UNK_PUNCT = 3, 48, 6

ALPHABET = "αβγδεζηθικλμνξοπρστυφχψω"
LETTER_IDS = {c: i for i, c in enumerate(ALPHABET)}
ID2LETTER = np.array(list(ALPHABET))

_EXTRA_BASE = {
    "ς": "σ", "ϲ": "σ", "Ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϕ": "φ", "ϰ": "κ", "ϱ": "ρ", "ϖ": "π",
}
_MARK_MAP = {
    0x0301: "acute", 0x0341: "acute", 0x0300: "grave", 0x0340: "grave",
    0x0342: "circ", 0x0302: "circ", 0x0313: "smooth", 0x0343: "smooth",
    0x0314: "rough", 0x0345: "iota", 0x0308: "diaer",
}
_ACC = {"acute": 1, "grave": 2, "circ": 3}
_BR = {"smooth": 1, "rough": 2}
_PUNCT_CLASS = {",": 1, "·": 2, "·": 2, ":": 3}   # small punct-plane LUT (comma/mid-dot/colon);
                                                    # sentence-final .;!? are captured via the
                                                    # boundary plane instead, same convention as
                                                    # processing_char_bert.py's base processor.


def _pack_dia(acc, br, iota, diaer):
    return ((acc * 3 + br) * 2 + iota) * 2 + diaer


# ==== Universal Dependencies conversion ========================================================
#
# The model's native output is the Perseus/AGDT annotation scheme its OGA training treebank
# uses: a 9-position morphological tag (pos.person.number.tense.mood.voice.gender.case.degree,
# one letter per position, "-" for n/a) and Prague-style dependency relations (SBJ, OBJ, ATR,
# ADV, PNOM, PRED, APOS, COORD, Aux*, ATV/AtvV, ExD, MWE, OCOMP) -- NOT Universal Dependencies.
#
# This is a solved, documented conversion problem, not something designed from scratch here:
# both UD_Ancient_Greek-Perseus and UD_Ancient_Greek-PROIEL are themselves automatic
# conversions of exactly this annotation style, via Zeman & Ramasamy's HamleDT pipeline
# (Zeman et al. 2014, "HamleDT: Harmonized Multi-Language Dependency Treebank", LRE) --
# specifically Treex::Block::HamleDT::{GRC::Harmonize,HarmonizePerseus,Udep} and
# Treex::Tool::PhraseBuilder::PragueToUD (github.com/ufal/treex), plus the morphology decoder
# Lingua::Interset::Tagset::GRC::Conll (github.com/dan-zeman/interset). The tables and
# restructuring rules below port that design to our exact tag/relation inventory.
#
# Two relations require real TREE restructuring, not just a label rename, because UD attaches
# function words/conjuncts differently than Prague style:
#   - COORD: Prague style makes the coordinating conjunction the head, conjuncts its children.
#     UD makes the FIRST conjunct the head, the conjunction becomes a `cc` dependent of it, and
#     other conjuncts become `conj` dependents.
#   - AuxP: Prague style makes the preposition the head of its complement. UD reverses this:
#     the nominal complement becomes the head, the preposition becomes a `case` dependent of it.
#   - Pnom (predicate nominal / copula): Prague style makes the copula verb the head with the
#     nominal attached as Pnom. UD makes the NOMINAL the head, the copula a `cop` dependent,
#     and reattaches the subject to the nominal instead of the verb.
#
# What relation does the PROMOTED node (first conjunct / nominal complement) get in UD, given
# it used to be a plain dependent of COORD/AuxP? In full AGDT/HamleDT, the coordinator's or
# preposition's own outbound relation carries the phrase's true external function via a
# _CO/_AP label suffix (e.g. "Sb_Co"), which HamleDT strips off and hands to the promoted node.
# Our 24-label inventory has no such suffix mechanism -- verified directly against real training
# examples in oga_sota.conllu (e.g. "ἀπὸ" and "ἕως" each attach to their external governor with
# deprel=AuxP verbatim, and "καὶ" attaches with deprel=COORD verbatim, regardless of whether the
# phrase functions as a locative adjunct, a coordinated subject, or anything else). So the label
# itself never carries a recoverable function here, and the promoted node gets UD's generic
# fallback instead of a falsely precise guess: "obl" for AuxP nominals, "dep" for COORD first
# conjuncts (see Stage 1/2 below).
#
# SCOPE: this ports the core, well-documented HamleDT/UD restructuring rules and a straight
# per-label relabeling for everything else. It does NOT replicate every Ancient-Greek-specific
# refinement in the full pipeline (e.g. GRC::Harmonize's regex-based negation-particle
# splitting, or the is_member/shared-modifier _CO/_AP distinction discussed above) -- treat this
# as a faithful, reasonably-scoped port, not a byte-exact reproduction of the official
# UD_Ancient_Greek-Perseus release.

# position 1 (POS) -> UD UPOS. 'c' defaults to CCONJ, refined to SCONJ if its converted deprel
# is "mark" (i.e. it was AuxC); 'v' stays VERB regardless of mood (participles etc. get
# VerbForm=Part as a feature, not a UPOS change, matching standard UD_Ancient_Greek practice).
_POS_TO_UPOS = {
    "a": "ADJ", "c": "CCONJ", "d": "ADV", "g": "PART", "i": "INTJ", "l": "DET",
    "m": "NUM", "n": "NOUN", "p": "PRON", "r": "ADP", "u": "PUNCT", "v": "VERB", "x": "X",
}
_NUMBER = {"d": "Dual", "p": "Plur", "s": "Sing"}
_TENSE_ASPECT = {   # (Tense, Aspect-or-None); aorist/perfect/future-perfect analysis per
    "a": ("Past", "Perf"),   # aorist = perfective past
    "f": ("Fut", None),
    "i": ("Past", "Imp"),    # imperfect
    "l": ("Pqp", None),      # pluperfect
    "p": ("Pres", None),
    "r": ("Pres", "Perf"),   # perfect = present relevance of a completed action
    "t": ("Fut", "Perf"),    # future perfect
}
_MOOD_VERBFORM = {   # (Mood-or-None, VerbForm)
    "i": ("Ind", "Fin"), "s": ("Sub", "Fin"), "o": ("Opt", "Fin"), "m": ("Imp", "Fin"),
    "n": (None, "Inf"), "p": (None, "Part"), "g": (None, "Gdv"),
}
_VOICE = {"a": "Act", "m": "Mid", "p": "Pass", "e": "Mid"}   # e = mediopassive, approximated as Mid
_GENDER = {"c": "Com", "f": "Fem", "m": "Masc", "n": "Neut"}
_CASE = {"a": "Acc", "d": "Dat", "g": "Gen", "n": "Nom", "v": "Voc"}
_DEGREE = {"c": "Cmp", "s": "Sup"}


def xpos_to_upos_feats(xpos: str) -> tuple[str, dict]:
    """9-position Perseus/AGDT tag (e.g. "v3spia---") -> (UD UPOS, UD FEATS dict).
    "-" in any position means not applicable/unspecified and is simply omitted from FEATS."""
    pos = xpos[0] if xpos else "x"
    upos = _POS_TO_UPOS.get(pos, "X")
    feats = {}
    rest = xpos[1:].ljust(8, "-")
    person, number, tense, mood, voice, gender, case, degree = rest[:8]
    if person in "123":
        feats["Person"] = person
    if number in _NUMBER:
        feats["Number"] = _NUMBER[number]
    if tense in _TENSE_ASPECT:
        t, a = _TENSE_ASPECT[tense]
        feats["Tense"] = t
        if a:
            feats["Aspect"] = a
    if mood in _MOOD_VERBFORM:
        m, vf = _MOOD_VERBFORM[mood]
        if m:
            feats["Mood"] = m
        feats["VerbForm"] = vf
    if voice in _VOICE:
        feats["Voice"] = _VOICE[voice]
    if gender in _GENDER:
        feats["Gender"] = _GENDER[gender]
    if case in _CASE:
        feats["Case"] = _CASE[case]
    if degree in _DEGREE:
        feats["Degree"] = _DEGREE[degree]
    return upos, feats


# Straight per-label relabeling for everything that ISN'T COORD/AuxP/Pnom (those need the tree
# restructuring in to_ud() below). Homoglyph artifacts in the training treebank (Greek "Ζ"/"Κ"
# instead of Latin "Z"/"K" in a couple of AuxZ/AuxK instances) are normalized first.
_HOMOGLYPH_FIX = {"AuxΖ": "AuxZ", "AuxΚ": "AuxK"}
_SIMPLE_DEPREL = {
    "SBJ": "nsubj", "OBJ": "obj", "OCOMP": "obj", "APOS": "appos",
    "AuxC": "mark", "AuxV": "aux", "AuxG": "punct", "AuxK": "punct", "AuxX": "punct",
    "AuxR": "expl", "AuxY": "cc", "AuxZ": "advmod", "AuxΖ": "advmod", "AuxΚ": "punct",
    "ATV": "xcomp", "AtvV": "xcomp", "ExD": "orphan", "MWE": "fixed",
    "PRED": "root",
}


def _normalize_deprel(label: str | None) -> str | None:
    return _HOMOGLYPH_FIX.get(label, label)


def to_ud(sent: list[dict]) -> list[dict]:
    """Convert one sentence's native decode() output (list of {form, xpos, lemma, upos, head,
    deprel} dicts, 1-indexed word order, head=0 meaning root) into UD-style output: each dict
    becomes {form, lemma, upos (UD), xpos (native tag, kept as-is per UD convention that XPOS
    is language-specific and optional), feats (dict), head, deprel (UD)}.

    See the module-level comment above for the restructuring rules and scope/limitations.
    """
    n = len(sent)
    if n == 0:
        return []
    heads = [0] * (n + 1)
    deprels = [None] * (n + 1)
    native_upos = [None] * (n + 1)
    for i, w in enumerate(sent, start=1):
        heads[i] = w["head"] if w["head"] is not None else 0
        deprels[i] = _normalize_deprel(w["deprel"])
        native_upos[i] = w["upos"]

    def children_of():
        c = {i: [] for i in range(0, n + 1)}
        for i in range(1, n + 1):
            c.setdefault(heads[i], []).append(i)
        return c

    # Stage 1: COORD -> first-conjunct-head restructuring.
    for c in [i for i in range(1, n + 1) if deprels[i] == "COORD"]:
        kids = sorted(children_of().get(c, []))
        if not kids:
            continue
        first, rest = kids[0], kids[1:]
        heads[first] = heads[c]
        deprels[first] = "dep"   # COORD's own outbound label carries no recoverable function; see above
        heads[c] = first
        deprels[c] = "cc"
        for k in rest:
            heads[k] = first
            if deprels[k] not in ("AuxG", "AuxK", "AuxX", "AuxΖ", "AuxΚ"):  # not a punctuation child
                deprels[k] = "conj"
            # punctuation children just reattach to `first`, keeping their own (later-relabeled) deprel

    # Stage 2: AuxP -> case restructuring (nominal complement becomes the head).
    for p in [i for i in range(1, n + 1) if deprels[i] == "AuxP"]:
        kids = sorted(children_of().get(p, []))
        if not kids:
            continue
        nominal, others = kids[0], kids[1:]
        heads[nominal] = heads[p]
        deprels[nominal] = "obl"   # AuxP's own outbound label carries no recoverable function; see above
        heads[p] = nominal
        deprels[p] = "case"
        for o in others:
            heads[o] = nominal

    # Stage 3: Pnom (predicate nominal) -> copula reversal. The nominal becomes the head in
    # place of the copula verb; the verb becomes `cop`; anything else attached to the verb
    # (chiefly the subject) is reattached to the nominal.
    for nom in [i for i in range(1, n + 1) if deprels[i] == "PNOM"]:
        verb = heads[nom]
        if verb == 0:
            continue
        heads[nom] = heads[verb]
        deprels[nom] = deprels[verb] if deprels[verb] != "PRED" else "root"
        heads[verb] = nom
        deprels[verb] = "cop"
        for i in range(1, n + 1):
            if i != nom and heads[i] == verb:
                heads[i] = nom

    # Stage 4: everything else -- straight relabel, with a few UPOS/context-sensitive refinements.
    for i in range(1, n + 1):
        d = deprels[i]
        if d in ("cc", "conj", "case", "cop", "dep", "obl"):     # already converted above
            continue
        if d == "ATR":
            deprels[i] = ("det" if native_upos[i] == "l"
                          else "amod" if native_upos[i] == "a"
                          else "nmod")
        elif d == "ADV":
            deprels[i] = ("advmod" if native_upos[i] == "d"
                          else "advcl" if native_upos[i] == "v" else "obl")
        elif d in _SIMPLE_DEPREL:
            deprels[i] = _SIMPLE_DEPREL[d]
        # else: already UD-style (e.g. a second pass over an already-converted label) or
        # unrecognized -- left as-is rather than silently dropped.

    out = []
    for i, w in enumerate(sent, start=1):
        upos, feats = xpos_to_upos_feats(w["xpos"])
        if deprels[i] == "mark" and upos == "CCONJ":
            upos = "SCONJ"   # refine c (CCONJ default) once we know it introduced a subordinate clause
        out.append(dict(form=w["form"], lemma=w["lemma"], upos=upos, xpos=w["xpos"],
                         feats=feats, head=heads[i], deprel=deprels[i]))
    return out


# ---- edit-script machinery (tagger/edits.py, ported verbatim on strings) ---------------------

GRAVE, ACUTE = "̀", "́"
_STRIP_MARKS = {"̄", "̆", "̣"}   # macron, breve, dot-below


def form_key(s: str) -> str:
    """Fold a surface form for script/lexicon keys: lowercase, grave->acute, strip
    macron/breve/underdot. NFC output. (tagger.edits.form_key, verbatim.)"""
    s = unicodedata.normalize("NFD", s.lower())
    s = s.replace(GRAVE, ACUTE)
    s = "".join(ch for ch in s if ch not in _STRIP_MARKS)
    return unicodedata.normalize("NFC", s)


def apply_script(form: str, sc) -> str | None:
    """tagger.edits.apply_script, verbatim. sc = (p_cut, p_add, s_cut, s_add, cap)."""
    p_cut, p_add, s_cut, s_add, _cap = sc
    if len(form) < p_cut + s_cut:
        return None
    return p_add + form[p_cut:len(form) - s_cut or None] + s_add


# ---- per-word character classification (subset of processing_char_bert.py's _classify) -------


def classify_word(word: str):
    """NFD-normalize `word`; return (chars, dia, cap) int lists, or None if it has no Greek
    letters (e.g. pure punctuation / digits / Latin -- these get no word slot, matching
    tagger/dataset.py's encode_word/encode_sentence convention)."""
    nfd = unicodedata.normalize("NFD", word)
    chars, dia, cap = [], [], []
    acc = br = iota = diaer = 0
    for ch in nfd:
        low = ch.lower()
        base = low if low in LETTER_IDS else _EXTRA_BASE.get(low)
        if base is not None:
            chars.append(LETTER_IDS[base])
            cap.append(1 if ch != low else 0)
            dia.append(0)
            acc = br = iota = diaer = 0
        elif unicodedata.combining(ch) or ord(ch) in _MARK_MAP:
            kind = _MARK_MAP.get(ord(ch))
            if kind in _ACC:
                acc = _ACC[kind]
            elif kind in _BR:
                br = _BR[kind]
            elif kind == "iota":
                iota = 1
            elif kind == "diaer":
                diaer = 1
            if dia:
                dia[-1] = _pack_dia(acc, br, iota, diaer)
        # anything else inside a single pre-tokenized word (stray punctuation) is dropped
    if not chars:
        return None
    return chars, dia, cap


def punct_class(tok: str) -> int:
    """Punctuation class a non-Greek token contributes to the preceding word's punct plane."""
    return max((_PUNCT_CLASS.get(c, 0) for c in tok), default=0)


def encode_sentence(tokens: list[str]):
    """tokens: pre-tokenized FORMs (CoNLL-U-style; punctuation already its own element).
    Returns (chars, boundary, dia, punct, cap, word_id, forms) for one packed sentence-row.
    `forms` holds the surface FORM of each *encodable* (has-Greek-letters) token, in word-slot
    order -- needed later to apply the lemma edit-script back onto real text."""
    chars, boundary, dia, punct, cap, word_id = [], [], [], [], [], []
    forms = []
    w = 0
    for tok in tokens:
        enc = classify_word(tok)
        if enc is None:
            pc = punct_class(tok)
            if pc and punct:
                punct[-1] = max(punct[-1], pc)
            continue
        c, d, cp = enc
        n = len(c)
        chars.extend(c)
        dia.extend(d)
        cap.extend(cp)
        punct.extend([0] * n)
        boundary.extend([0] * (n - 1) + [1])
        word_id.extend([w] * n)
        forms.append(tok)
        w += 1
    if boundary:
        boundary[-1] = 2   # sentence end
    return chars, boundary, dia, punct, cap, word_id, forms


_TRAIL_PUNCT_RE = re.compile(r"^(.*?)([.;!?,·:]+)$")


class CharBertJointProcessor:
    """`processor(sentences)` -> dict of batched tensors ready for
    `CharBertForTaggingAndParsing(**batch)`; `processor.decode(model_out, batch)` -> per-sentence
    list of word-level (form, xpos, lemma, upos, head, deprel) records."""

    def __init__(self, vocab: dict | None = None, deprel_vocab: list | None = None):
        self.vocab = vocab or {}
        self.deprel_vocab = deprel_vocab or []
        self.scripts = [tuple(s[:4]) + (bool(s[4]),) for s in self.vocab.get("scripts", [])]
        self.xpos_alpha = self.vocab.get("xpos_alpha", [])
        self.tags = self.vocab.get("tags", [])
        self.upos = self.vocab.get("upos", [])
        self.lex_ft = self.vocab.get("lex_ft", {})
        self.lex_f = self.vocab.get("lex_f", {})
        self.nongreek = self.vocab.get("nongreek", {})
        self.fallback_xpos = self.vocab.get("fallback_xpos", "u--------")
        self.fallback_upos = self.vocab.get("fallback_upos", "u")

        if self.xpos_alpha and self.tags:
            xid = [{c: i for i, c in enumerate(a)} for a in self.xpos_alpha]
            xpos_len = len(self.xpos_alpha)
            self.tag_idx = torch.tensor(
                [[xid[p].get(t[p] if p < len(t) else "-", 0) for p in range(xpos_len)]
                 for t in self.tags],
                dtype=torch.long,
            )
        else:
            self.tag_idx = None
        self._script_pc = np.array([s[0] + s[2] for s in self.scripts]) if self.scripts else None
        self._app_cache = {}

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_pretrained(cls, path, **_kwargs):
        path = Path(path)
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        deprel_vocab = json.loads((path / "deprel_vocab.json").read_text(encoding="utf-8"))
        return cls(vocab, deprel_vocab)

    def save_pretrained(self, save_directory, **_kwargs):
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        (save_directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False))
        (save_directory / "deprel_vocab.json").write_text(json.dumps(self.deprel_vocab, ensure_ascii=False))
        (save_directory / "processor_config.json").write_text(
            json.dumps({"processor_class": "CharBertJointProcessor"}))

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Naive plain-text -> token-list convenience: whitespace split + peel a run of
        trailing sentence/clause punctuation off each whitespace-chunk into its own token.
        Good enough for a short usage-example sentence; real corpora should be tokenized
        properly (this is not a substitute for that)."""
        out = []
        for chunk in text.strip().split():
            m = _TRAIL_PUNCT_RE.match(chunk)
            if m and m.group(1):
                out.append(m.group(1))
                out.extend(list(m.group(2)))
            else:
                out.append(chunk)
        return out

    # ---------------------------------------------------------------- encode

    def __call__(self, sentences: list[str] | list[list[str]]):
        if not sentences:
            raise ValueError("sentences must be non-empty")
        if isinstance(sentences[0], str):
            sentences = [sentences]   # single sentence -> batch of 1

        encs = [encode_sentence(s) for s in sentences]
        B = len(encs)
        Tmax = max(len(e[0]) for e in encs)
        Tmax = max(Tmax, 1)

        ids = np.full((B, Tmax), PAD_ID, dtype=np.int64)
        bnd = np.zeros((B, Tmax), dtype=np.int64)
        dia = np.zeros((B, Tmax), dtype=np.int64)
        pct = np.zeros((B, Tmax), dtype=np.int64)
        cp = np.zeros((B, Tmax), dtype=np.int64)
        seg = np.zeros((B, Tmax), dtype=np.int64)
        wid = np.full((B, Tmax), -1, dtype=np.int64)
        all_forms = []
        for b, (c, bd, d, p, ca, wi, forms) in enumerate(encs):
            n = len(c)
            if n:
                ids[b, :n] = c
                bnd[b, :n] = bd
                dia[b, :n] = d
                pct[b, :n] = p
                cp[b, :n] = ca
                seg[b, :n] = 1
                wid[b, :n] = wi
            all_forms.append(forms)

        batch = dict(
            input_ids=torch.from_numpy(ids),
            boundary=torch.from_numpy(bnd),
            dia=torch.from_numpy(dia),
            punct=torch.from_numpy(pct),
            cap=torch.from_numpy(cp),
            seg_id=torch.from_numpy(seg),
            word_id=torch.from_numpy(wid),
        )
        batch["_forms"] = all_forms   # kept out-of-band: not a model input, needed for decoding
        return batch

    # ---------------------------------------------------------------- decode: xpos / upos

    def decode_xpos(self, xpos_logits, word_mask, flat_logits=None):
        """Constrained decode: combine the 9 factored heads' log-probs (+ the flat full-tag
        head's log-probs, if present) over the attested-tag inventory, argmax. Mirrors
        tagger.decode.TagDecoder.xpos exactly."""
        assert self.tag_idx is not None, "vocab.json (xpos_alpha/tags) required for xpos decode"
        ti = self.tag_idx.to(xpos_logits[0].device)
        score = 0
        for p, lg in enumerate(xpos_logits):
            lp = torch.log_softmax(lg.float(), -1)
            score = score + lp[:, :, ti[:, p]]
        if flat_logits is not None:
            score = score + torch.log_softmax(flat_logits.float(), -1)
        best = score.argmax(-1).cpu()
        return [[self.tags[int(best[b, w])]
                 for w in range(word_mask.shape[1]) if word_mask[b, w]]
                for b in range(word_mask.shape[0])]

    def decode_upos(self, upos_logits, word_mask):
        pred = upos_logits.argmax(-1).cpu()
        return [[self.upos[int(pred[b, w])]
                 for w in range(word_mask.shape[1]) if word_mask[b, w]]
                for b in range(word_mask.shape[0])]

    # ---------------------------------------------------------------- decode: lemma

    def _applicable(self, L):
        if L not in self._app_cache:
            self._app_cache[L] = self._script_pc <= L
        return self._app_cache[L]

    def decode_lemma(self, form: str, script_logprobs: torch.Tensor, xpos: str | None = None,
                      use_lexicon: bool = True, topk: int = 64) -> str:
        """script_logprobs: (n_scripts,) log-softmax for this word. Mirrors
        tagger.decode.LemmaDecoder.__call__ exactly (lexicon-rescored edit script, with an
        open-script fallback for OOV forms)."""
        key = form_key(form)
        copy_cap = form[:1] != form[:1].lower()

        app = torch.from_numpy(self._applicable(len(key))).to(script_logprobs.device)
        masked = script_logprobs.masked_fill(~app, -1e30)
        k = min(topk, masked.shape[-1])
        topv, topi = masked.topk(k)
        topv, topi = topv.tolist(), topi.tolist()

        if use_lexicon:
            cands = None
            if xpos is not None:
                cands = self.lex_ft.get(key + "\t" + xpos)
            if not cands:
                cands = self.lex_f.get(key, {})
            cands = dict(cands)
            if cands:
                lower = {}
                for lemma in cands:
                    lower.setdefault(lemma.lower(), lemma)
                best, best_s = None, -1e30
                for s, i in zip(topv, topi):
                    if s <= -1e29:
                        break
                    out = apply_script(key, self.scripts[i])
                    lemma = lower.get(out)
                    if lemma is not None:
                        sc = s + 1e-3 * np.log1p(cands[lemma])
                        if sc > best_s:
                            best, best_s = lemma, sc
                return best if best is not None else max(cands, key=cands.get)

        sc = self.scripts[topi[0]]
        lemma = apply_script(key, sc)
        if lemma is None:
            return form
        if sc[4] or copy_cap:
            lemma = lemma[:1].upper() + lemma[1:]
        return lemma

    # ---------------------------------------------------------------- decode: dependency arcs

    @staticmethod
    def decode_arcs(arc_scores, rel_scores):
        """Greedy per-token argmax head (0=root) + label argmax at the chosen head. Mirrors
        parser.biaffine.BiaffineHead.decode exactly."""
        heads_out = arc_scores.argmax(-1)
        B, W = heads_out.shape
        bi = torch.arange(B, device=arc_scores.device)[:, None].expand(B, W)
        wi = torch.arange(W, device=arc_scores.device)[None, :].expand(B, W)
        labels_out = rel_scores[bi, wi, heads_out].argmax(-1)
        return heads_out.cpu(), labels_out.cpu()

    # ---------------------------------------------------------------- full decode

    def decode(self, model_out, batch, use_lexicon: bool = True, ud: bool = False) -> list[list[dict]]:
        """model_out: a CharBertJointOutput (or the equivalent return_dict=False tuple).
        batch: the dict returned by __call__ (needs `_forms`).
        -> per sentence, a list of dicts, word order.

        Default (ud=False): the model's NATIVE Perseus/AGDT-style output --
        {form, xpos, lemma, upos, head, deprel}, where `upos` is the Perseus single-letter
        code (not UD) and `deprel` is a Prague-style relation (SBJ/OBJ/ATR/COORD/AuxP/...).

        ud=True: Universal Dependencies-style output instead -- {form, lemma, upos (UD),
        xpos (native tag, kept as-is per UD convention), feats (dict), head, deprel (UD)}.
        See `to_ud()` for the conversion (a documented, but not 100%-official-treebank-exact,
        port of the HamleDT/UD_Ancient_Greek-Perseus conversion methodology)."""
        word_mask = model_out.word_mask if hasattr(model_out, "word_mask") else model_out[-1]
        xpos_logits = model_out.xpos_logits if hasattr(model_out, "xpos_logits") else model_out[0]
        script_logits = model_out.script_logits if hasattr(model_out, "script_logits") else model_out[1]
        upos_logits = model_out.upos_logits if hasattr(model_out, "upos_logits") else model_out[2]
        flat_logits = model_out.flat_logits if hasattr(model_out, "flat_logits") else model_out[3]
        arc_scores = model_out.arc_scores if hasattr(model_out, "arc_scores") else model_out[4]
        rel_scores = model_out.rel_scores if hasattr(model_out, "rel_scores") else model_out[5]

        xpos_pred = self.decode_xpos(xpos_logits, word_mask, flat_logits=flat_logits)
        upos_pred = self.decode_upos(upos_logits, word_mask)
        script_logprobs = torch.log_softmax(script_logits.float(), -1)   # (B,W,n_script)

        heads_out, labels_out = (None, None)
        if arc_scores is not None:
            heads_out, labels_out = self.decode_arcs(arc_scores, rel_scores)

        forms_batch = batch["_forms"]
        results = []
        for b, forms in enumerate(forms_batch):
            n = len(forms)
            sent_out = []
            for w in range(n):
                form = forms[w]
                xpos = xpos_pred[b][w]
                upos = upos_pred[b][w]
                lemma = self.decode_lemma(form, script_logprobs[b, w], xpos=xpos, use_lexicon=use_lexicon)
                head = int(heads_out[b, w]) if heads_out is not None else None
                deprel = (self.deprel_vocab[int(labels_out[b, w])]
                          if labels_out is not None and self.deprel_vocab else None)
                sent_out.append(dict(form=form, xpos=xpos, lemma=lemma, upos=upos,
                                      head=head, deprel=deprel))
            results.append(to_ud(sent_out) if ud else sent_out)
        return results
