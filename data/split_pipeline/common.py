"""Shared normalization / tokenization / hashing for the 10-fold split pipeline.

The matching key ("skeleton") is designed so that differences between editions
(diacritics, breathings, iota subscript, sigma forms, editorial brackets,
punctuation, casing, line breaks) are invisible to the matcher.
"""
import unicodedata
import regex as re
import xxhash

# ---------------------------------------------------------------- zones
# zone codes stored per record
ZONE_B0 = 0            # rotating literary buckets B0..B9 -> codes 0..9
ZONE_PTEST = 10        # PHI/TM number ends in 3 (fixed test, all folds)
ZONE_PVAL = 11         # PHI/TM number ends in 4 (fixed val, all folds)
ZONE_TRAIN = 12        # train in every fold (repaired, bronze, digit 0-2,5-9)

N_BUCKETS = 10
NGRAM = 8              # word n-gram size for contamination matching AND
                       # MinHash shingles (user-approved strictness level:
                       # any shared normalized 8-gram = contamination)
MIN_SEG_CHARS = 100    # min chars for a stitched train segment
MASK_PTEST = 1 << 10
MASK_PVAL = 1 << 11


def fold_conflict_mask(k):
    """Zones that are val/test in fold k -> bitmask a train sentence must avoid."""
    return (1 << k) | (1 << ((k + 1) % N_BUCKETS)) | MASK_PTEST | MASK_PVAL


# ---------------------------------------------------------------- skeleton
_COMBINING = re.compile(r"\p{Mn}+")
# keep only Greek-script LETTERS: two passes (drop non-Greek, then drop
# Greek-script non-letters such as keraia / numeral signs)
_NON_GREEK = re.compile(r"[^\p{Script=Greek}]+")
_NON_LETTER = re.compile(r"[^\p{L} ]+")
_SIGMA = str.maketrans({"ς": "σ", "ϲ": "σ", "Ϲ": "σ", "ϐ": "β", "ϑ": "θ", "ϰ": "κ"})


def skeleton(text):
    """Normalized matching key: diacritic-free lowercase Greek words, space-sep."""
    t = unicodedata.normalize("NFD", text)
    t = _COMBINING.sub("", t)
    t = t.lower().translate(_SIGMA)
    t = _NON_GREEK.sub(" ", t)
    t = _NON_LETTER.sub(" ", t)
    return " ".join(t.split())


# ---------------------------------------------------------------- sentences
# sentence terminators: period, Greek/Latin question marks, exclamation,
# ano teleia U+0387, middle dot U+00B7, semicolon (Greek question mark shares
# the codepoint in many editions), U+037E, and blank lines / colon.
_SENT_BOUNDARY = re.compile("[.;!?:\u00b7\u0387\u037e]+|\\n\\s*\\n")


def sentence_spans(text):
    """Split into sentence spans [(start, end)) covering the whole string.

    The terminator and following whitespace belong to the preceding sentence,
    so concatenating consecutive spans reproduces the original text exactly.
    Spans whose skeleton is empty are merged into nothing (skipped) but their
    characters stay attached to the previous span to keep full coverage.
    """
    spans = []
    prev = 0
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.end()
        spans.append((prev, end))
        prev = end
    if prev < len(text):
        spans.append((prev, len(text)))
    # attach empty-skeleton spans to their predecessor (or successor)
    out = []
    for s, e in spans:
        if skeleton(text[s:e]):
            out.append([s, e])
        elif out:
            out[-1][1] = e
        else:
            out.append([s, e])  # leading junk span; may still have empty skeleton
    # drop a leading span with empty skeleton by merging into the next
    if len(out) > 1 and not skeleton(text[out[0][0]:out[0][1]]):
        out[1][0] = out[0][0]
        out.pop(0)
    return [(s, e) for s, e in out]


# ---------------------------------------------------------------- grouping
def work_prefix(source, rid):
    """Volume/work granularity used for clustering and sibling-zone lookup.

    Deliberately NOT source-qualified: the same archive.org volume appears
    as source greek_pd in pristine and source ia in repaired -- the bare
    volume/work identifier must match across sources and tiers.
    """
    if source == "oga":
        return ".".join(rid.split(".")[:2])       # tlgXXXX.tlgYYY work urn
    return rid.split("#")[0]


# ---------------------------------------------------------------- hashing
def h64(s):
    return xxhash.xxh64_intdigest(s)


def sent_hashes(skel):
    """(exact_hash, bag_hash, word_list) for one sentence skeleton."""
    words = skel.split()
    exact = h64(skel)
    bag = h64(" ".join(sorted(words)))
    return exact, bag, words


def ngram_hashes(words, n=NGRAM):
    """Hashes of all word n-grams of a word list (empty if fewer than n words)."""
    if len(words) < n:
        return []
    return [h64(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)]
