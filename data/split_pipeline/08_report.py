#!/usr/bin/env python3
"""Stage 8: write README.md + stats.md into the output directory."""
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.expandvars(os.environ.get("FOLD_OUTDIR", "$STOICHEIA_DATA"))
W = os.path.join(ROOT, "work")


def load(name):
    p = os.path.join(W, name)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


s1, s2, s3 = load("stage1_stats.json"), load("stage2_stats.json"), load("stage3_stats.json")
s4, s5, s6 = load("stage4_stats.json"), load("stage5_stats.json"), load("stage6_stats.json")
s3b, s11 = load("stage3b_stats.json"), load("stage11_stats.json")
dc, s12 = load("doublecheck_report.json"), load("stage12_stats.json")
ver = load("verify_report.json")

TIERS = ["pristine", "repaired", "bronze", "inscriptions"]

lines = ["# 10-fold split — statistics", ""]

lines += ["## Verification (independent re-check of materialized folds)", "",
          "| fold | train vs val+test 8-gram hits | sentence hits | val vs test 8-gram | sentence | status |",
          "|---|---|---|---|---|---|"]
for k in range(10):
    v = ver.get("fold_%d" % k)
    if not v:
        lines.append("| %d | – | – | – | – | not verified |" % k)
        continue
    t, vv = v["train_vs_valtest"], v["val_vs_test"]
    lines.append("| %d | %d | %d | %d | %d | %s |" % (
        k, t["gram_hits"], t["sent_hits"], vv["gram_hits"], vv["sent_hits"],
        "PASS" if v["PASS"] else "**FAIL**"))
lines.append("")

if s6:
    lines += ["## Per-fold output (records / Mchars emitted; Mchars excised; records dropped)", ""]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    for key, v in s6.items():
        k, split, tier = key.split("|")
        a = agg[int(k)][(split, tier)]
        for j in range(4):
            a[j] += v[j]
    for k in range(10):
        lines.append("### fold %d" % k)
        lines.append("| split | tier | records | Mchars | Mchars excised | dropped records |")
        lines.append("|---|---|---|---|---|---|")
        for split in ("train", "val", "test"):
            for tier in TIERS:
                r = agg[k].get((split, tier))
                if not r:
                    continue
                lines.append("| %s | %s | %d | %.1f | %.1f | %d |" % (
                    split, tier, r[0], r[1] / 1e6, r[2] / 1e6, r[3]))
        lines.append("")

if dc:
    lines += ["## Adversarial double-check", "",
              "- D1 mutation test (verifier detects planted verbatim AND"
              " orthographically-mutated editions; flags nothing on genuine"
              " train): **%s**" % ("PASS" if dc.get("D1_mutation", {}).get("PASS") else "FAIL"),
              "- D2 TM/PHI digit-rule compliance + D3 id-disjointness per fold: **%s**"
              % ("PASS" if all(v.get("PASS") for k, v in dc.items()
                               if k.startswith("fold_")) else "FAIL"), ""]
if s11:
    tot = sum(v["dropped_bronze"] for v in s11.values()) // max(1, len(s11))
    lines += ["## Greek-origin bronze exclusion (anti-paraphrase guard)", "",
              "~%d bronze records (back-translations of Latin translations of"
              " GREEK works: Graeca miscellanea, Biblia, Ptolemaeus Latinus,"
              " Versiones latinae, Greek-author works in PL etc.) removed from"
              " every fold's train." % tot, ""]
if s12:
    lines += ["## Rare-word paraphrase screen (per-fold bronze filter)", "",
              "| fold | bronze dropped | Mchars |", "|---|---|---|"]
    for k in range(10):
        v = s12.get("fold_%d" % k, {})
        lines.append("| %d | %s | %s |" % (k, v.get("dropped_bronze", "–"),
                                           v.get("dropped_Mchars", "–")))
    lines.append("")

for name, s in [("Stage 1 (tokenization)", s1), ("Stage 2 (pristine clustering)", s2),
                ("Stage 3 (bucket packing)", s3), ("Stage 3b (canonical zones)", s3b),
                ("Stage 4 (contamination index)", s4),
                ("Stage 5 (conflict masks)", s5)]:
    lines += ["## " + name, "", "```json", json.dumps(s, indent=2), "```", ""]

with open(os.path.join(OUTDIR, "stats.md"), "w") as f:
    f.write("\n".join(lines))

readme = """# 10-fold leakage-proof split of the Ancient Greek corpora

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

Pipeline: $STOICHEIA_ROOT/pipeline/
"""
with open(os.path.join(OUTDIR, "README.md"), "w") as f:
    f.write(readme)
print("wrote", os.path.join(OUTDIR, "README.md"), "and stats.md")
