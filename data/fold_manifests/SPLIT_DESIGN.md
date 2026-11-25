# 10-fold leakage-proof split of the Ancient Greek corpora

Built from:
- `Ericu950/AncientGreek` (pristine + repaired tiers)
- `Ericu950/SyntheticAncientGreek-CorpusCorporum` (bronze)
- `Ericu950/Inscriptions_2` (PHI inscriptions + GPT-4o synthetic variants)

## Layout
`fold_k/{train,val,test}.jsonl.zst` for k = 0..9.

Records: `{"id", "tier", "source", "text"}`; excised train records are split
into `id#segN` segments (maximal runs of consecutive clean sentences).
Val/test inscriptions instead carry the full original PHI row **minus** the
synthetic fields.

## Fold design
- Literary pristine was clustered (same text in multiple editions -> one
  cluster: id-prefix, exact-duplicate, MinHash-LSH and shared-sentence
  evidence) and clusters packed into 10 buckets of ~10% of words each.
  Fold k: test = bucket k, val = bucket (k+1) mod 10, train = the rest.
- Papyri (ddbdp via TM number, dclp) and PHI inscriptions use the fixed
  Ithaca-compatible digit rule in every fold: number ends in 3 -> test,
  4 -> val, else train. Synthetic variants of a val/test PHI number appear
  nowhere.
- repaired + bronze are train-only. A repaired record additionally inherits
  the ZONE of its pristine sibling volume/work (same id prefix) or, for
  papyri, its TM digit -- so OCR-damaged copies and sibling pages of a
  val/test document are excluded from that fold's train entirely, not just
  excised sentence-wise.
- Bronze passages that are back-translations of Latin translations of
  GREEK-origin works are removed from train in every fold: they are machine
  paraphrases of real Greek texts and would be invisible to verbatim
  matching. Two layers: (1) catalog flags -- translation corpora (Graeca
  miscellanea, Biblia, Ptolemaeus Latinus, Versiones latinae), Greek-writing
  authors, Greek author names or translation markers in the TITLE (catches
  e.g. "Iliados liber XIV Latine redditus" by 'Anonymus'); (2) an empirical
  rare-word screen per fold -- any remaining bronze record sharing >= 4 rare
  words (test-side document frequency <= 3, length >= 6) with a single test
  record is dropped from that fold's train, deliberately over-excluding
  topical coincidences. The remaining bronze derives from genuinely Latin
  works, whose Greek is model-generated language with no Greek source text
  to memorize.

## Intended claim
A model trained on fold k's train set has never seen the test texts of fold
k in any form: not the same edition, not another edition, not OCR-damaged or
LLM-repaired copies, not reordered or unpunctuated variants, not quotations
of 8+ words, not whole sentences of any length, not GPT-4o synthetic
variants of test inscriptions, and not machine back-translations of Latin
translations of Greek test works. Conjectures the model produces on test
texts are therefore grounded in the rest of Greek literature, not in
memorization of the target text. Residual channels that remain by design:
sub-8-word formulaic phrases (shared language, not text identity), human
paraphrase within Greek literature itself (scholia, indirect tradition --
the same evidence a human conjecturer legitimately uses), and other works
by the same author.

## Decontamination (100% strict, no frequency exemptions)
All text is normalized to a matching skeleton (NFD, diacritics stripped,
lowercase, sigma-folded, non-Greek-letters removed) and sentence-tokenized.
A train sentence is EXCISED when, against any zone that is val/test in the
fold, it has (a) an identical skeleton, (b) an identical sorted-word bag, or
(c) any shared word-8-gram (computed over cross-sentence word streams), or
(d) its record matches a val/test-eligible pristine record at document level
(MinHash est. Jaccard >= 0.5, both records >= 35 words, sizes within 3x). Remaining consecutive sentences are stitched
into segments (>= 100 chars, train; >= 25 chars, val). Val is additionally
cleaned against test the same way.

Guarantee (verified independently per fold, see stats.md): train shares no
8-word verbatim sequence (after normalization) and no complete sentence with
val or test; val shares none with test. Sub-8-word fragments inside longer
differing sentences are not treated as leakage. Bronze is machine
back-translation from Latin; verbatim quotations are caught, free paraphrase
of content is out of scope by design.

Pipeline: `data/split_pipeline/`
