"""Build the two tokenizers of the representation ablation from the SAME rendered text.

  char  flat Unicode characters (NFC codepoints incl. accented/capital letters, punctuation) with a
        Metaspace word marker "▁"; no merges, so every character is one token.
  bpe   byte-pair encoding, 32K vocabulary, same Metaspace pre-tokenization.

Special ids are identical in both: [PAD]=0 [MASK]=1 [BLANK]=2 [UNK]=3.

Training text: a fixed random sample of documentary-free pretraining records (pristine, repaired and
bronze tiers in the 1 : 1 : 0.3 stable-phase proportion), normalized and rendered through the plane
codec (ablate/render.py), so tokenizers see exactly what the token arms train on.

  python -m ablate.build_tokenizers --shards $STOICHEIA_DATA/shards --out $STOICHEIA_DATA/ablate/tok
"""
from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SPECIALS = ["[PAD]", "[MASK]", "[BLANK]", "[UNK]"]
DOC_SOURCES = {"ddbdp", "dclp"}


def sample_texts(shards: Path, n_records: int, seed: int = 0):
    from ablate.render import render
    rng = np.random.default_rng(seed)
    per = {"pristine": n_records * 1.0 / 2.3, "repaired": n_records * 1.0 / 2.3,
           "bronze": n_records * 0.3 / 2.3}
    for sub, tiers in (("v1_punct", ("pristine", "repaired")), ("bronze_punct", ("bronze",))):
        d = shards / sub
        idx = pq.read_table(d / "index.parquet")
        offs = idx.column("offset").to_numpy(); lens = idx.column("length").to_numpy()
        tier = idx.column("tier").to_numpy(zero_copy_only=False)
        src = idx.column("source").to_numpy(zero_copy_only=False)
        planes = {p: np.memmap(d / f"{p}.bin", dtype=np.uint8, mode="r")
                  for p in ("chars", "boundary", "dia", "cap", "punct")}
        for t in tiers:
            el = np.flatnonzero((tier == t) & ~np.isin(src, list(DOC_SOURCES)) & (lens >= 20))
            pick = rng.choice(el, size=min(int(per[t]), len(el)), replace=False)
            for i in np.sort(pick):
                o, l = int(offs[i]), int(min(lens[i], 8192))
                txt, _ = render(*(np.asarray(planes[p][o:o + l])
                                  for p in ("chars", "boundary", "dia", "cap", "punct")))
                yield txt


def build(texts, out: Path, bpe_vocab: int, min_char_count: int):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    out.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    (out / "sample_stats.json").write_text(json.dumps(dict(
        records=len(texts), chars=sum(map(len, texts)))))

    def fast(tok, name):
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
        f = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", mask_token="[MASK]",
                                    unk_token="[UNK]", additional_special_tokens=["[BLANK]"])
        f.save_pretrained(out / name)
        return f

    # ---- char: alphabet = every character seen >= min_char_count times, plus the word marker
    cnt = Counter()
    for t in texts:
        cnt.update(t.replace(" ", "▁"))
    cnt["▁"] += len(texts)
    alphabet = sorted(c for c, k in cnt.items() if k >= min_char_count)
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for c in alphabet:
        vocab.setdefault(c, len(vocab))
    char_tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    fc = fast(char_tok, "char")

    # ---- bpe
    bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
    bpe.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.BpeTrainer(vocab_size=bpe_vocab, special_tokens=SPECIALS,
                                  initial_alphabet=alphabet, limit_alphabet=len(alphabet),
                                  min_frequency=2, show_progress=False)
    bpe.train_from_iterator(texts, trainer=trainer)
    fb = fast(bpe, "bpe")

    probe = texts[:2000]
    stats = {}
    for name, f in (("char", fc), ("bpe", fb)):
        ids = f(probe, add_special_tokens=False)["input_ids"]
        n_tok = sum(map(len, ids)); n_chr = sum(map(len, probe))
        n_unk = sum(x == f.unk_token_id for s in ids for x in s)
        stats[name] = dict(vocab=len(f), tokens_per_char=round(n_tok / n_chr, 4),
                           unk_rate=round(n_unk / max(n_tok, 1), 6))
        assert f.convert_tokens_to_ids(SPECIALS) == [0, 1, 2, 3], name
    (out / "tokenizer_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-records", type=int, default=400_000)
    ap.add_argument("--bpe-vocab", type=int, default=32_000)
    ap.add_argument("--min-char-count", type=int, default=50)
    a = ap.parse_args()
    build(sample_texts(Path(a.shards), a.n_records), Path(a.out), a.bpe_vocab, a.min_char_count)


if __name__ == "__main__":
    main()
