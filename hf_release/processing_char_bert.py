"""HF-Hub-compatible processor for Stoicheia: text <-> the model's four input planes.

Wraps the reference normalization/denormalization logic (character classification,
diacritic packing, word/sentence-boundary detection) into a single callable that
produces model-ready tensors, plus decode helpers for the three masking use cases
described in the model card (restoration, accent recovery, re-segmentation).

This is intentionally NOT a `PreTrainedTokenizer` subclass: the underlying encoding is
a row-per-letter, four-parallel-plane structure (not a single token-id stream), which
doesn't fit that base class's assumptions. It follows the same `register_for_auto_class`
mechanism transformers uses for tokenizers/feature extractors, so
`AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)` works the same way.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MASK, BLANK, PAD = 24, 25, 26
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
_MARK_CHARS = {"acute": "́", "grave": "̀", "circ": "͂",
               "smooth": "̓", "rough": "̔", "iota": "ͅ", "diaer": "̈"}
# punct plane classes (matches data/normalize.py's canonical 6-way scheme):
# 0 none, 1 comma, 2 high-dot(·), 3 colon, 4 period, 5 question/exclamation
_PUNCT_CHARS = {1: ",", 2: "·", 3: ":", 4: ".", 5: ";"}

# "λόγ[5±3]καὶ" -- a lacuna of uncertain width: best guess 5 letters, plausible
# range 5-3..5+3. See CharBertProcessor.restore_elastic.
_ELASTIC_RE = re.compile(r"\[(\d+)±(\d+)\]")


def _pack_dia(acc, br, iota, diaer):
    return ((acc * 3 + br) * 2 + iota) * 2 + diaer


def _unpack_dia(d):
    diaer = d % 2; d //= 2
    iota = d % 2; d //= 2
    br = d % 3; acc = d // 3
    return acc, br, iota, diaer


@dataclass
class _Encoded:
    chars: list  # int, may include MASK
    boundary: list
    dia: list
    punct: list
    cap: list  # original capitalization, for round-tripping non-masked positions


class CharBertProcessor:
    """`processor(text)` -> dict of batched tensors ready for `CharBertModel(**batch)`."""

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def save_pretrained(self, save_directory, **_kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / "processor_config.json").write_text(json.dumps({"processor_class": "CharBertProcessor"}))

    # ---------------------------------------------------------------- encode

    def _classify(self, text: str) -> _Encoded:
        """Turn raw NFC/NFD polytonic text into per-letter plane lists, damage ('-' runs)
        preserved as MASK/UNK positions, gold values kept for every other position."""
        nfd = unicodedata.normalize("NFD", text)
        chars, boundary, dia, punct, cap = [], [], [], [], []
        acc = br = iota = diaer = 0
        pending_bnd = 0
        i = 0
        while i < len(nfd):
            ch = nfd[i]
            if ch == "-":
                run = 0
                while i < len(nfd) and nfd[i] == "-":
                    run += 1
                    i += 1
                for _ in range(run):
                    chars.append(MASK); boundary.append(UNK_BND)
                    dia.append(UNK_DIA); punct.append(UNK_PUNCT); cap.append(0)
                continue
            low = ch.lower()
            base = low if low in LETTER_IDS else _EXTRA_BASE.get(low)
            if base is not None:
                if chars and pending_bnd:
                    boundary[-1] = pending_bnd
                    pending_bnd = 0
                chars.append(LETTER_IDS[base])
                cap.append(1 if ch != low else 0)
                boundary.append(0)
                dia.append(0)  # filled in by trailing combining marks below
                punct.append(0)
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
            elif ch.isspace():
                pending_bnd = max(pending_bnd, 1)
            elif ch in ".;!?":
                pending_bnd = max(pending_bnd, 2)
                if punct:
                    punct[-1] = 4 if ch == "." else 5
            elif ch in ",:··":
                if punct:
                    punct[-1] = {",": 1, "·": 2, "·": 2, ":": 3}.get(ch, 0)
            i += 1
        if boundary:
            boundary[-1] = max(boundary[-1], 2)
        return _Encoded(chars, boundary, dia, punct, cap)

    def __call__(self, text: str, mask_planes: list[str] | None = None, has_boundaries: bool = True):
        """Encode `text` into model-ready tensors.

        mask_planes: any subset of {"chars", "boundary", "dia", "punct"} to force to
            UNKNOWN at every position (in addition to any '-' runs, which are always
            treated as a damaged/masked span regardless of mask_planes).
        has_boundaries: set False for scriptio continua input (no real spaces) so the
            boundary plane starts fully UNKNOWN rather than "no boundaries found".
        """
        mask_planes = set(mask_planes or [])
        enc = self._classify(text)
        n = len(enc.chars)
        chars = np.array(enc.chars, dtype=np.int64)
        boundary = np.array(enc.boundary, dtype=np.int64)
        dia = np.array(enc.dia, dtype=np.int64)
        punct = np.array(enc.punct, dtype=np.int64)

        if "chars" in mask_planes:
            chars[:] = MASK
        if "boundary" in mask_planes or not has_boundaries:
            boundary[:] = UNK_BND
        if "dia" in mask_planes:
            dia[:] = UNK_DIA
        if "punct" in mask_planes:
            punct[:] = UNK_PUNCT

        batch = dict(
            input_ids=torch.from_numpy(chars)[None],
            boundary=torch.from_numpy(boundary)[None],
            dia=torch.from_numpy(dia)[None],
            punct=torch.from_numpy(punct)[None],
            seg_id=torch.zeros(1, n, dtype=torch.long),
        )
        batch["_cap"] = enc.cap  # kept out-of-band; not a model input (cap is output-only)
        return batch

    # ---------------------------------------------------------------- decode

    @staticmethod
    def _restore_polytonic(chars, dia, cap, boundary, punct=None) -> str:
        words, cur = [], []
        for i in range(len(chars)):
            ch = ID2LETTER[chars[i]] if chars[i] < 24 else "?"
            a, b, io, dd = _unpack_dia(int(dia[i]))
            if cap[i]:
                ch = ch.upper()
            s = ch
            if b:
                s += _MARK_CHARS[{1: "smooth", 2: "rough"}[b]]
            if dd:
                s += _MARK_CHARS["diaer"]
            if a:
                s += _MARK_CHARS[{1: "acute", 2: "grave", 3: "circ"}[a]]
            if io:
                s += _MARK_CHARS["iota"]
            cur.append(s)
            p = int(punct[i]) if punct is not None else 0
            if boundary[i] >= 1:
                w = "".join(cur)
                if w and w[-1] == "σ":
                    w = w[:-1] + "ς"
                w = unicodedata.normalize("NFC", w)
                if p in _PUNCT_CHARS:
                    w += _PUNCT_CHARS[p]
                elif boundary[i] == 2:
                    w += "."
                words.append(w)
                cur = []
        if cur:
            w = unicodedata.normalize("NFC", "".join(cur))
            p = int(punct[-1]) if punct is not None else 0
            if p in _PUNCT_CHARS:
                w += _PUNCT_CHARS[p]
            words.append(w)
        return " ".join(words)

    def decode_restoration(self, model_out, batch, predict_punct: bool = True,
                           sentence_breaks: bool = True, gap_word_breaks: bool = True) -> str:
        """Fill masked positions with the model's argmax predictions; keep every
        other position exactly as given. Each plane is filled independently
        wherever IT is unknown -- chars/cap only inside a '-' gap (chars==MASK),
        but boundary/dia/punct wherever THAT plane is UNK, which may be the whole
        sequence if mask_planes was also used for joint gap+accent+boundary
        restoration (not just the '-' gap itself).

        `predict_punct=False` leaves punctuation exactly as given instead of filling
        it from the model. Documentary fine-tunes are trained on inscriptions and
        papyri, whose editions carry almost no punctuation, so their punctuation head
        is weakly supervised and tends to sprinkle stops into an otherwise correct
        reading; epigraphic and papyrological use generally wants it off.

        `sentence_breaks=False` demotes every *predicted* sentence boundary to a plain
        word boundary, so a filled gap comes back as running text. The same fine-tunes
        read editions in which sentence division is editorial rather than attested, and
        will happily place a full stop inside a word they otherwise restore correctly.

        `gap_word_breaks=False` forbids new word division *inside* a filled gap, so the
        restored letters continue the surrounding word. This is the common editorial
        case -- a break within a single word, as in `στεφά--- ἀρετῆς` -- where the model
        recovers the letters correctly but the boundary head, which is free to segment
        anywhere, may cut them into pieces. Leave it on when the lacuna plausibly spans
        a word boundary."""
        pred_char = model_out.char.argmax(-1)[0].tolist()
        pred_bnd = model_out.boundary.argmax(-1)[0].tolist()
        pred_dia = model_out.dia.argmax(-1)[0].tolist()
        pred_cap = model_out.cap.argmax(-1)[0].tolist()
        pred_punct = model_out.punct.argmax(-1)[0].tolist()
        chars = batch["input_ids"][0].tolist()
        boundary = batch["boundary"][0].tolist()
        dia = batch["dia"][0].tolist()
        punct = batch["punct"][0].tolist()
        cap = batch["_cap"]
        was_masked = [c == MASK for c in chars]
        for i in range(len(chars)):
            if chars[i] == MASK:
                chars[i] = pred_char[i] if pred_char[i] < 24 else 0
                cap[i] = pred_cap[i]
            if boundary[i] == UNK_BND:
                boundary[i] = 0 if (not gap_word_breaks and was_masked[i]) else pred_bnd[i]
                if not sentence_breaks and boundary[i] == 2:
                    boundary[i] = 1
            if dia[i] == UNK_DIA:
                dia[i] = pred_dia[i]
            if punct[i] == UNK_PUNCT:
                punct[i] = pred_punct[i] if predict_punct else 0
        return self._restore_polytonic(chars, dia, cap, boundary, punct)

    def decode_diacritics(self, model_out, batch) -> str:
        """Replace the diacritic plane with the model's predictions; letters/boundaries/
        capitalization/punctuation are taken from the input as given."""
        pred_dia = model_out.dia.argmax(-1)[0].tolist()
        chars = batch["input_ids"][0].tolist()
        boundary = batch["boundary"][0].tolist()
        punct = batch["punct"][0].tolist()
        cap = batch["_cap"]
        return self._restore_polytonic(chars, pred_dia, cap, boundary, punct)

    def decode_boundaries(self, model_out, batch) -> str:
        """Replace the boundary plane with the model's predictions (0/1/2); letters/
        diacritics/capitalization/punctuation are taken from the input as given.
        Only useful when the input truly has no accents either (a spaced-out or
        scriptio-continua text that already carries accents gives the boundary head
        a strong shortcut -- each word carries exactly one accent -- so this isn't a
        meaningful standalone test of the boundary head specifically; see
        decode_restoration/restore_elastic for the realistic joint case)."""
        pred_bnd = model_out.boundary.argmax(-1)[0].tolist()
        chars = batch["input_ids"][0].tolist()
        dia = batch["dia"][0].tolist()
        punct = batch["punct"][0].tolist()
        cap = batch["_cap"]
        return self._restore_polytonic(chars, dia, cap, pred_bnd, punct)

    def restore_respaced(self, model, text: str, **kw) -> str:
        """Restore a gap, then re-decide word division on the completed text.

        The five planes are predicted independently in one pass, which is fine when the
        whole context is bare (everything is decided together) but unreliable when a gap
        sits inside already-spaced text: the boundary head sees a half-known segmentation
        and hedges, so a correctly restored word can come back cut in two.

        This does it in the order an editor would: fill the letters first, throw away the
        spacing entirely, and run the model again over the resulting *scriptio continua*
        with the boundary and diacritic planes unknown everywhere -- the regime the model
        was pretrained on. `text` may carry either a `-` run or a `[N±M]` marker.
        """
        if _ELASTIC_RE.search(text):
            filled, _, _ = self.restore_elastic(model, text, **kw)
        else:
            batch = self(text)
            with torch.no_grad():
                out = model(**{k: v for k, v in batch.items() if not k.startswith("_")})
            filled = self.decode_restoration(out, batch, **kw)
        letters = "".join(ch for ch in unicodedata.normalize("NFD", filled)
                          if unicodedata.category(ch).startswith("L"))
        batch = self(letters, mask_planes=["dia", "boundary"], has_boundaries=False)
        with torch.no_grad():
            out = model(**{k: v for k, v in batch.items() if not k.startswith("_")})
        return self.decode_restoration(out, batch)

    def restore_elastic(self, model, text: str, min_width: int = 1,
                         mask_dia_boundary: bool = False, predict_punct: bool = True,
                         sentence_breaks: bool = True, gap_word_breaks: bool = True):
        """Restore a lacuna of *uncertain* width -- the realistic editorial case,
        since editors estimate a lacuna's length, they rarely know it exactly.

        `text` must contain exactly one `[N±M]` marker (best-guess width N,
        plausible range N-M..N+M), e.g. `"λόγ[5±3]καὶ ὁ λόγος ἦν πρὸς τὸν θεόν"`.
        For every candidate width in that range, this fills the gap, then scores
        each candidate by the mean log-probability of the model's own letter
        predictions inside the gap specifically (that's what distinguishes widths).

        `mask_dia_boundary` controls what happens OUTSIDE the gap:
          - False (default): real accents/word-boundaries already present in
            `text` are kept as given -- only the gap itself is filled. Use this
            for text where the surrounding context is already known/accented (the
            common editorial case: a lacuna in an otherwise-legible inscription).
          - True: accents and word-boundaries are masked and reconstructed
            everywhere, not just inside the gap -- for fully bare scriptio
            continua surrounding the lacuna too (no spaces, no accents at all).

        Returns `(best_text, best_width, candidates)`, where `candidates` is every
        `(width, filled_text, mean_logp)` tried, sorted best-first. Needs the model
        (not just its output), since it runs one forward pass per candidate width.
        """
        m = _ELASTIC_RE.search(text)
        if not m:
            raise ValueError("text must contain one '[N±M]' marker, e.g. 'λόγ[5±3]καὶ'")
        n, spread = int(m.group(1)), int(m.group(2))
        prefix, suffix = text[:m.start()], text[m.end():]
        gap_start = len(self._classify(prefix).chars)

        candidates = []
        for L in range(max(min_width, n - spread), n + spread + 1):
            probe = prefix + ("-" * L) + suffix
            if mask_dia_boundary:
                batch = self(probe, mask_planes=["dia", "boundary"], has_boundaries=False)
            else:
                batch = self(probe)
            with torch.no_grad():
                out = model(**{k: v for k, v in batch.items() if not k.startswith("_")})

            logp = torch.log_softmax(out.char, dim=-1)[0]
            pred_char = out.char.argmax(-1)[0].tolist()
            gap_logp = sum(logp[gap_start + i, pred_char[gap_start + i]].item()
                            for i in range(L)) / L

            pred_bnd = out.boundary.argmax(-1)[0].tolist()
            pred_dia = out.dia.argmax(-1)[0].tolist()
            pred_cap = out.cap.argmax(-1)[0].tolist()
            pred_punct = out.punct.argmax(-1)[0].tolist()
            chars = batch["input_ids"][0].tolist()
            boundary = batch["boundary"][0].tolist()
            dia = batch["dia"][0].tolist()
            punct = batch["punct"][0].tolist()
            cap = batch["_cap"]
            was_masked = [c == MASK for c in chars]
            for i in range(len(chars)):
                if chars[i] == MASK:
                    chars[i] = pred_char[i] if pred_char[i] < 24 else 0
                    cap[i] = pred_cap[i]
                if boundary[i] == UNK_BND:
                    boundary[i] = 0 if (not gap_word_breaks and was_masked[i]) else pred_bnd[i]
                    if not sentence_breaks and boundary[i] == 2:
                        boundary[i] = 1
                if dia[i] == UNK_DIA:
                    dia[i] = pred_dia[i]
                if punct[i] == UNK_PUNCT:
                    punct[i] = pred_punct[i] if predict_punct else 0
            filled = self._restore_polytonic(chars, dia, cap, boundary, punct)
            candidates.append((L, filled, gap_logp))

        candidates.sort(key=lambda c: -c[2])
        best_L, best_text, _ = candidates[0]
        return best_text, best_L, candidates
