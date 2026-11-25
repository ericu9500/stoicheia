"""Treebank -> model batches.

Each syntactic word's FORM is encoded independently through Stoicheia's
normalize_record (guaranteeing exact word<->char-span alignment), sentences are the
concatenation of their encodable words, and whole sentences are greedily packed into
fixed-length rows with per-sentence seg_ids (block-diagonal attention, exactly like
pretraining's document packing). All input planes carry their true values — chars,
boundary (word/sentence ends), dia, punct — since all of them are known from raw text
at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from tagger.backbone import Stats, normalize_record
from tagger.edits import compute_script, form_key

# punctuation class LUT from the pretraining normalizer (comma/high-dot/colon/period/question)
from data.normalize import _PUNCT as PUNCT_LUT  # noqa: E402

PAD_ID = 26


def encode_word(form: str):
    """(chars, dia, cap) uint8 arrays for one FORM, or None if it has no Greek letters."""
    r = normalize_record(form, Stats(), with_punct=True)
    if r is None:
        return None
    chars, _boundary, dia, cap, _punct = r
    return chars, dia, cap


def punct_class(form: str) -> int:
    """Punctuation class a non-Greek token contributes to the preceding word."""
    return max((int(PUNCT_LUT[ord(c)]) for c in form if ord(c) < len(PUNCT_LUT)), default=0)


@dataclass
class SentEnc:
    chars: np.ndarray
    boundary: np.ndarray
    dia: np.ndarray
    punct: np.ndarray
    cap: np.ndarray
    spans: list          # per token: (start, end) char span or None (unencodable)
    y_xpos: np.ndarray   # (n_enc_words, 9) int64, -100 = unseen-in-train
    y_script: np.ndarray # (n_enc_words,)
    y_upos: np.ndarray   # (n_enc_words,)
    y_tag: np.ndarray    # (n_enc_words,) full-XPOS-tag id

    def __len__(self):
        return len(self.chars)


def encode_sentence(sent, vocab=None) -> SentEnc | None:
    """vocab=None -> encode inputs only (labels filled with -100)."""
    parts, spans = [], []
    n = 0
    for t in sent.tokens:
        enc = encode_word(t.form)
        if enc is None:
            spans.append(None)
            # non-Greek token: contribute its punctuation class to the previous word
            if parts:
                pc = punct_class(t.form)
                if pc:
                    parts[-1]["punct"][-1] = max(parts[-1]["punct"][-1], pc)
            continue
        chars, dia, cap = enc
        parts.append(dict(chars=chars, dia=dia, cap=cap,
                          boundary=np.zeros(len(chars), dtype=np.uint8),
                          punct=np.zeros(len(chars), dtype=np.uint8), tok=t))
        parts[-1]["boundary"][-1] = 1
        spans.append((n, n + len(chars)))
        n += len(chars)
    if not parts:
        return None
    parts[-1]["boundary"][-1] = 2   # sentence end

    labs = np.full((len(parts), 12), -100, dtype=np.int64)
    if vocab is not None:
        for i, p in enumerate(parts):
            t = p["tok"]
            labs[i, :9] = vocab.xpos_ids(t.xpos)
            labs[i, 9] = vocab.script_id(compute_script(form_key(t.form), t.lemma))
            labs[i, 10] = vocab.upos_id(t.upos)
            labs[i, 11] = vocab.tag_id(t.xpos)

    return SentEnc(
        chars=np.concatenate([p["chars"] for p in parts]),
        boundary=np.concatenate([p["boundary"] for p in parts]),
        dia=np.concatenate([p["dia"] for p in parts]),
        punct=np.concatenate([p["punct"] for p in parts]),
        cap=np.concatenate([p["cap"] for p in parts]),
        spans=spans,
        y_xpos=labs[:, :9], y_script=labs[:, 9], y_upos=labs[:, 10], y_tag=labs[:, 11],
    )


@dataclass
class Row:
    """One packed model row plus everything needed to map predictions back."""
    sents: list = field(default_factory=list)   # (sent_index, SentEnc)


def pack_rows(encs, T=2048, W=384, order=None):
    """Greedy packing of whole sentences (in `order`) into rows of <=T chars, <=W words.
    Oversize sentences are truncated to T at a word boundary (span-less tail words fall
    back to the lexicon rule at decode time); truncation count is returned for logging."""
    order = range(len(encs)) if order is None else order
    rows, truncated = [], 0
    cur, cur_c, cur_w = Row(), 0, 0
    for si in order:
        e = encs[si]
        if e is None:
            continue
        nc, nw = len(e), len(e.y_script)
        if nc > T or nw > W:
            truncated += 1
            continue   # pathological; handled by rule fallback at decode time
        if cur_c + nc > T or cur_w + nw > W:
            rows.append(cur)
            cur, cur_c, cur_w = Row(), 0, 0
        cur.sents.append((si, e))
        cur_c += nc
        cur_w += nw
    if cur.sents:
        rows.append(cur)
    return rows, truncated


def batch_rows(rows, T=2048, W=384, device=None):
    """Stack a list of Rows into model tensors + label tensors + slot metadata.

    Returns dict with input_ids/boundary/dia/punct/seg_id (B,T), word_id (B,T) in
    [-1,W), y_xpos (B,W,9), y_script (B,W), y_upos (B,W), and slots: per row, a list
    of (sent_index, token_index) per word slot (for mapping predictions back).
    """
    B = len(rows)
    ids = np.full((B, T), PAD_ID, dtype=np.int64)
    bnd = np.zeros((B, T), dtype=np.int64)
    dia = np.zeros((B, T), dtype=np.int64)
    pct = np.zeros((B, T), dtype=np.int64)
    cp = np.zeros((B, T), dtype=np.int64)
    seg = np.zeros((B, T), dtype=np.int64)
    wid = np.full((B, T), -1, dtype=np.int64)
    y = np.full((B, W, 12), -100, dtype=np.int64)
    slots = []
    for b, row in enumerate(rows):
        c = w = 0
        rs = []
        for k, (si, e) in enumerate(row.sents):
            n = len(e)
            ids[b, c:c + n] = e.chars
            bnd[b, c:c + n] = e.boundary
            dia[b, c:c + n] = e.dia
            pct[b, c:c + n] = e.punct
            cp[b, c:c + n] = e.cap
            seg[b, c:c + n] = k + 1
            j = 0
            for ti, span in enumerate(e.spans):
                if span is None:
                    continue
                s0, s1 = span
                wid[b, c + s0:c + s1] = w
                y[b, w, :9] = e.y_xpos[j]
                y[b, w, 9] = e.y_script[j]
                y[b, w, 10] = e.y_upos[j]
                y[b, w, 11] = e.y_tag[j]
                rs.append((si, ti))
                w += 1
                j += 1
            c += n
        slots.append(rs)
    t = lambda a: torch.from_numpy(a) if device is None else torch.from_numpy(a).to(device)
    return dict(input_ids=t(ids), boundary=t(bnd), dia=t(dia), punct=t(pct), cap=t(cp),
                seg_id=t(seg),
                word_id=t(wid), y_xpos=t(y[:, :, :9]), y_script=t(y[:, :, 9]),
                y_upos=t(y[:, :, 10]), y_tag=t(y[:, :, 11]), slots=slots)


@dataclass
class HFSentEnc:
    """One sentence's HF subword encoding: real tokenizer ids for the WHOLE sentence text
    (Greek and non-Greek tokens alike -- a subword LM was pretrained on running text and should
    see punctuation etc. as context), plus a word_id-style alignment and the same label arrays
    encode_sentence produces, in the same order (only "encodable" = has-Greek-letters tokens,
    per encode_word, get a pooled word slot / a label row -- exactly the CharBERT convention, so
    XPOS/script/UPOS/lemma-edit-script targets and parser.model.build_gold's gold-arc indexing
    line up 1:1 across both backbones)."""
    input_ids: list
    word_id: list           # length == len(input_ids); slot in [0, n_enc) or -1 (incl. specials
                             # and non-Greek tokens, which get real subwords but no word slot)
    enc_orig_idx: list       # original sent.tokens index for each of the n_enc word slots, in
                             # slot order -- mirrors the char path's (sent_index, token_index)
                             # bookkeeping in `slots` for build_gold / JointModel._regroup
    y_xpos: np.ndarray
    y_script: np.ndarray
    y_upos: np.ndarray
    y_tag: np.ndarray

    def __len__(self):
        return len(self.input_ids)


def encode_sentence_hf(sent, tokenizer, vocab=None, max_len=512):
    """HF subword tokenization + word alignment for one sentence, or None if it has no
    encodable (Greek) tokens, or if the untruncated sequence exceeds max_len subword positions
    (dropped whole, like pack_rows' oversize-sentence rule for the char path -- no partial/
    misaligned sentences)."""
    words = [t.form for t in sent.tokens]
    if not words:
        return None
    enc_idx = [i for i, t in enumerate(sent.tokens) if encode_word(t.form) is not None]
    if not enc_idx:
        return None
    slot_of = {orig: k for k, orig in enumerate(enc_idx)}

    labs = np.full((len(enc_idx), 12), -100, dtype=np.int64)
    if vocab is not None:
        for k, i in enumerate(enc_idx):
            t = sent.tokens[i]
            labs[k, :9] = vocab.xpos_ids(t.xpos)
            labs[k, 9] = vocab.script_id(compute_script(form_key(t.form), t.lemma))
            labs[k, 10] = vocab.upos_id(t.upos)
            labs[k, 11] = vocab.tag_id(t.xpos)

    tok_out = tokenizer(words, is_split_into_words=True)
    ids = tok_out["input_ids"]
    if len(ids) > max_len:
        return None
    wraw = tok_out.word_ids()
    wid = [slot_of.get(w, -1) if w is not None else -1 for w in wraw]

    return HFSentEnc(input_ids=ids, word_id=wid, enc_orig_idx=enc_idx,
                      y_xpos=labs[:, :9], y_script=labs[:, 9], y_upos=labs[:, 10],
                      y_tag=labs[:, 11])


def batch_sentences_hf(items, tokenizer, W=384, device=None):
    """items: list of (sent_index, HFSentEnc), one row per sentence -- ordinary padded batching
    (attention_mask) stands in for the char pipeline's block-diagonal packing, which existed
    only to make CharBERT's char-window local attention cheap; a standard HF encoder attends
    over the whole (padded) sentence and needs no such trick.

    Returns dict with input_ids/attention_mask (B,Tmax), word_id (B,Tmax) in [-1,W), y_xpos
    (B,W,9), y_script/y_upos/y_tag (B,W), and slots: per row, a list of (sent_index,
    token_index) per word slot -- same shape/semantics as batch_rows' `slots`.
    """
    B = len(items)
    Tmax = max(len(e) for _, e in items)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    ids = np.full((B, Tmax), pad_id, dtype=np.int64)
    attn = np.zeros((B, Tmax), dtype=np.int64)
    wid = np.full((B, Tmax), -1, dtype=np.int64)
    y = np.full((B, W, 12), -100, dtype=np.int64)
    slots = []
    for b, (si, e) in enumerate(items):
        n = len(e)
        ids[b, :n] = e.input_ids
        attn[b, :n] = 1
        wid[b, :n] = e.word_id
        n_enc = e.y_xpos.shape[0]
        y[b, :n_enc, :9] = e.y_xpos
        y[b, :n_enc, 9] = e.y_script
        y[b, :n_enc, 10] = e.y_upos
        y[b, :n_enc, 11] = e.y_tag
        slots.append([(si, ti) for ti in e.enc_orig_idx])
    t = lambda a: torch.from_numpy(a) if device is None else torch.from_numpy(a).to(device)
    return dict(input_ids=t(ids), attention_mask=t(attn), word_id=t(wid),
                y_xpos=t(y[:, :, :9]), y_script=t(y[:, :, 9]), y_upos=t(y[:, :, 10]),
                y_tag=t(y[:, :, 11]), slots=slots)


def pack_dev_items(encs, W, tokenizer=None, T=2048, order=None):
    """Row/item list for evaluation (unsharded; caller shards across ranks) or for one
    training epoch's shuffled pass. CharBERT path -> pack_rows' packed Rows (T-limited,
    block-diagonal); HF path -> a flat (sent_index, HFSentEnc) list, one row per sentence.
    Returns (rows_or_items, truncated_count)."""
    if tokenizer is None:
        return pack_rows(encs, T, W, order)
    order = range(len(encs)) if order is None else order
    items = [(i, encs[i]) for i in order if encs[i] is not None]
    return items, 0


def batch_chunk(chunk, T, W, tokenizer=None, device=None):
    """Stack a chunk of pack_dev_items' output into model tensors; dispatches on backbone kind
    exactly like pack_dev_items does."""
    if tokenizer is None:
        return batch_rows(chunk, T, W, device=device)
    return batch_sentences_hf(chunk, tokenizer, W, device=device)


class TaggerDataset:
    """Encodes a .conllu once; repacks (shuffled) per epoch.

    tokenizer=None (default) -> CharBERT char-plane pipeline (encode_sentence / pack_rows /
    batch_rows), unchanged. tokenizer=<a HF fast tokenizer> -> HF subword pipeline
    (encode_sentence_hf + batch_sentences_hf, one sentence per row, no T-limited packing)."""

    def __init__(self, sentences, vocab, T=2048, W=384, tokenizer=None, hf_max_len=512):
        self.T, self.W = T, W
        self.sentences = sentences
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.encs = [encode_sentence(s, vocab) for s in sentences]
        else:
            encs = [encode_sentence_hf(s, tokenizer, vocab, max_len=hf_max_len)
                    for s in sentences]
            # mirror pack_rows' oversize-sentence rule: drop (not truncate) sentences whose
            # encodable-word count can't fit a row of width W
            self.encs = [e if (e is not None and e.y_xpos.shape[0] <= W) else None for e in encs]
        self.n_enc = sum(e is not None for e in self.encs)

    def batches(self, micro, seed=None, shuffle=True):
        order = np.arange(len(self.encs))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        rows, _ = pack_dev_items(self.encs, self.W, self.tokenizer, self.T, order)
        for i in range(0, len(rows), micro):
            yield batch_chunk(rows[i:i + micro], self.T, self.W, self.tokenizer)
