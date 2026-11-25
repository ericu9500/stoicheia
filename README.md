# Stoicheia

A character-level masked-diffusion Transformer for Ancient Greek, pretrained on an
open, revision-pinned corpus and released as eleven decontaminated checkpoints (ten
rotated literary folds + one documentary-clean model), fine-tuned for restoration of
damaged inscriptions/papyri, morphosyntactic tagging and dependency parsing, and
macronization/metrical scansion.

This repository is the training/evaluation code. The pretrained and fine-tuned model
weights are on the HuggingFace Hub — see [`MODEL_CARDS_INDEX.md`](MODEL_CARDS_INDEX.md)
for the full list, or jump straight to
[`Ericu950/Stoicheia-doc_clean`](https://huggingface.co/Ericu950/Stoicheia-doc_clean)
(the flagship backbone) or
[`Ericu950/Stoicheia-restoration-test3`](https://huggingface.co/Ericu950/Stoicheia-restoration-test3) (or any of the ten digit-rotation checkpoints) /
[`-tagger-parser`](https://huggingface.co/Ericu950/Stoicheia-tagger-parser) for a
ready-to-use downstream model (or [`-meter`](https://huggingface.co/Ericu950/Stoicheia-meter) for
macronization and scansion). All model repos are public: weights ship as `model.safetensors` with a `config.json`,
loadable directly through `AutoModel.from_pretrained(..., trust_remote_code=True)`.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericu9500/stoicheia/blob/main/Stoicheia_demo.ipynb)

Run everything in the browser: [`Stoicheia_demo.ipynb`](Stoicheia_demo.ipynb) restores a lacuna in
unaccented text, picks the checkpoint that has provably never read your document, tags and parses a
verse of Homer, macronizes and scans a line, and scores the macronizer on the benchmark.

## Quickstart (no training required)

```python
import torch
from transformers import AutoModel
from huggingface_hub import hf_hub_download

REPO = "Ericu950/Stoicheia-doc_clean"
model = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()

# `trust_remote_code` loads the model classes; the processor is a separate helper, so
# fetch it into the working directory before importing it.
hf_hub_download(repo_id=REPO, filename="processing_char_bert.py", local_dir=".")
from processing_char_bert import CharBertProcessor

processor = CharBertProcessor()

# a lacuna in bare scriptio continua (no spaces, no accents), the usual case for damaged,
# unaccented primary sources. "[N±M]" restores the gap at every width in N-M..N+M, with
# accents and word division, and ranks the candidates by the model's confidence.
# Experimental: the models do not reliably infer the width of a gap (paper, Appendix F),
# so give the physical estimate where you have one. Inferring the extent of a lacuna is
# left to future work.
text = "εναρχηηνο[5±3]καιολογοςηνπροστονθεον"
best_text, best_width, candidates = processor.restore_elastic(model, text, mask_dia_boundary=True)
print(best_text)  # -> ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν.
```

A damaged inscription, unaccented and unspaced where the break falls:

```python
print(processor.restore_respaced(model, "ἔδοξεν τηβου-- καὶ τῷ δήμῳ"))
# -> ἔδοξεν τῇ βουλῇ καὶ τῷ δήμῳ
```

Accents and word division are predictions, not requirements: a bare majuscule transcript is as
readable to this model as a modern critical text, and the gap is filled in the same pass that
decides where the words end.


## What's here

- `model/`, `data/`, `train/`, `eval/` — the pretraining architecture (`CharBertEncoder`,
  a five-plane character-level masked-diffusion Transformer) and training loop.
- `insc/` — restoration fine-tuning (inscriptions + papyri) and strict-protocol
  evaluation (same-harness comparison against DeepMind's Ithaca).
- `tagger/`, `parser/` — morphosyntactic tagging (factored XPOS, edit-script lemma,
  UPOS) and biaffine dependency parsing, plus a joint multi-task model and a
  pluggable HuggingFace-encoder bridge for cross-encoder ablations.
- `meter/` — macronization (vowel length) and metrical scansion, including the
  *Norma* benchmark protocol and rule-based silver-data mining pipeline.
- `tests/` — CPU-only pytest suite.
- `scripts/fetch_dbbe.py` — refetches the Database of Byzantine Book Epigrams, which the
  released corpus omits. Run it to reconstruct the pretraining corpus exactly (5,476
  records, ~0.2M words, 0.1% of the total); the records come under DBBE's own terms.

See [`REPRODUCING.md`](REPRODUCING.md) for the full environment setup and end-to-end
reproduction walkthrough.

## Citation

```bibtex
@inproceedings{cullhed2027stoicheia,
  title     = {Stoicheia: Character-Level Masked Diffusion for Ancient Greek Textual
               Restoration, Parsing, and Metrical Scansion},
  author    = {Cullhed, Eric and Th\"orn Cleland, Albin},
  booktitle = {Proceedings of the Conference of the European Chapter of the
               Association for Computational Linguistics (EACL 2027)},
  year      = {2027}
}
```

## License

Apache 2.0 (see `LICENSE`). External baselines (DeepMind's Ithaca and predictingthepast releases) are
downloaded separately from their own repositories and retain their own licenses — see
`NOTICE` and `REPRODUCING.md`.
