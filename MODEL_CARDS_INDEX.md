# Stoicheia — Released Models and Datasets

All models are character-level masked-diffusion Transformers for Ancient Greek (405M
params, `d_model=1024`, depth 32), loadable via
`AutoModel.from_pretrained(repo_id, trust_remote_code=True)`. See each model card for
usage examples specific to that checkpoint's task.

## Pretrained backbones (11)

Ten rotated, work-level-decontaminated literary folds plus one documentary-clean model.
For any passage in the open training corpus, at least one of these eleven has provably
never seen it during pretraining.

| model | decontamination |
|---|---|
| [`Ericu950/Stoicheia-doc_clean`](https://huggingface.co/Ericu950/Stoicheia-doc_clean) | zero documentary (inscription/papyrus) exposure of any kind |
| [`Ericu950/Stoicheia-fold-0`](https://huggingface.co/Ericu950/Stoicheia-fold-0) … [`fold-9`](https://huggingface.co/Ericu950/Stoicheia-fold-9) | rotated 80/10/10 literary split, fold *k*'s test set unseen by fold *k*'s model |

## Fine-tuned downstream models (4)

Built on `Stoicheia-doc_clean`, except the macronizer (see its card).

| model | task |
|---|---|
| [`Ericu950/Stoicheia-restoration-test0`](https://huggingface.co/Ericu950/Stoicheia-restoration-test0) … [`-test9`](https://huggingface.co/Ericu950/Stoicheia-restoration-test9) | documentary restoration, ten checkpoints — checkpoint *k* holds out every PHI/TM identifier ending in digit *k*, so every document in the corpus has a model that provably never saw it |
| [`Ericu950/Stoicheia-tagger-parser`](https://huggingface.co/Ericu950/Stoicheia-tagger-parser) | morphosyntactic tagging (XPOS/UPOS/lemma) + dependency parsing |
| [`Ericu950/Stoicheia-meter`](https://huggingface.co/Ericu950/Stoicheia-meter) | macronization (vowel length) + metrical scansion, trained jointly |
| [`Ericu950/Stoicheia-macronizer`](https://huggingface.co/Ericu950/Stoicheia-macronizer) | macronization only: one fine-tuning of the paper's macron-only recipe (see its card for the backbone) |

## Datasets

| dataset | what it is |
|---|---|
| [`Ericu950/AncientGreek`](https://huggingface.co/datasets/Ericu950/AncientGreek) | pretraining corpus, ~361M words in `pristine` and `repaired` tiers |
| [`Ericu950/SyntheticAncientGreek-CorpusCorporum`](https://huggingface.co/datasets/Ericu950/SyntheticAncientGreek-CorpusCorporum) | bronze synthetic augmentation tier, machine-translated from Latin |
| [`Ericu950/Inscriptions_2`](https://huggingface.co/datasets/Ericu950/Inscriptions_2) | PHI inscriptions used for the 10-fold split and restoration fine-tuning |
| [`Ericu950/Stoicheia-meter-silver`](https://huggingface.co/datasets/Ericu950/Stoicheia-meter-silver) | silver macronization/scansion training data (Hypotactic-derived + constraint-solver mined) |
| [`Ericu950/norma`](https://huggingface.co/datasets/Ericu950/norma) | mirror of the *Norma Syllabarum Graecarum* benchmark, in the exact split the paper evaluates on |

`AncientGreek` omits the Database of Byzantine Book Epigrams; `scripts/fetch_dbbe.py`
refetches it under its own terms. Not re-released at all: the OGA/AGDT treebank splits, which come from Celano's
own repository and are cited rather than duplicated.
