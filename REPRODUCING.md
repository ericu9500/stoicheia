# Reproducing Stoicheia

## 1. Environment

```bash
git clone https://github.com/ericu9500/stoicheia
cd stoicheia
export STOICHEIA_DATA=/path/to/a/writable/data+checkpoints/directory
source env.sh
pip install -e .
```

For GPU training at scale, the project used an Apptainer container on aarch64 GH200
nodes via a SLURM cluster; the container image itself is not included in this repo (see
`scripts/slurm/README.md` for what's expected inside it, and how to point `STOICHEIA_SIF`
at your own equivalent, or run without a container at all). See the same file for the
cluster-specific sbatch templates (edit the account/partition placeholders before
submitting). For CPU-only work (tests, small-scale inference, data preparation),
`pip install -r requirements-cpu.txt` is sufficient — no container needed,
`attn_impl="sdpa"` runs on CPU.

## 2. Data

Three core datasets are already public and used as-is (no re-release needed):

```python
from datasets import load_dataset
gold_silver = load_dataset("Ericu950/AncientGreek")                         # pretraining corpus
bronze = load_dataset("Ericu950/SyntheticAncientGreek-CorpusCorporum")      # synthetic augmentation
inscriptions = load_dataset("Ericu950/Inscriptions_2")                     # PHI inscriptions
```

Two further data dependencies are external, citable resources — clone/download them
directly rather than expecting a copy in this repo:
- **OGA/AGDT treebank** (tagging/parsing fine-tuning, morphosyntax evaluation): clone
  Celano's own repository, `git.informatik.uni-leipzig.de/celano/morphosyntactic_parser_for_oga`.
- **Norma** (macronization/scansion benchmark): released with the vowel-length paper
  cited in the paper's macronization section; the harnesses read it through
  `MACRONIZER_SRC` (see the last section of this file).

The 10-fold decontamination split is built by the pipeline in `data/split_pipeline/`
(MinHash-LSH near-duplicate clustering, a last-digit rule for papyri/inscriptions,
and n-gram decontamination against the eval sets — see the last section of this file
and `data/fold_manifests/SPLIT_DESIGN.md`). Its downstream consumer,
`data/build_fold_shards.py`, takes a fold's `train.jsonl.zst` and builds the memmap
shards the pretraining loader reads. In practice you don't need to rebuild the fold
assignments from scratch: use the pretrained checkpoints directly
(`MODEL_CARDS_INDEX.md`), and `data/fold_manifests/` records every fold's test-work
assignments for verification.

## 3. Pretraining (the 11 backbones)

```bash
sbatch scripts/slurm/pretrain_fold.sbatch 0     # one of ten literary folds (0-9)
sbatch scripts/slurm/pretrain_doc_clean.sbatch  # the documentary-clean model
```

Each is a long-running, checkpointed, resumable job chain (dev-driven schedule, not
step-capped — see the paper's Model section for the staged-anneal training regime).
Pretrained checkpoints are also available directly on the Hub (see
`MODEL_CARDS_INDEX.md`) — you do not need to re-pretrain to use or fine-tune the models.

## 4. Fine-tuning the three downstream tasks

All three fine-tune from `$STOICHEIA_DATA/runs/stoicheia_doc_clean/best.pt` (or the equivalent
Hub checkpoint, downloaded locally first if you want to fine-tune outside this
pipeline's own checkpoint format).

```bash
# restoration, v2 recipe, held-out digit 3 (the paper's headline model)
sbatch scripts/slurm/insc_finetune_whole_4node.sbatch configs/insc/finetune_whole_v4_t3v4.json

# joint tagger + dependency parser
sbatch scripts/slurm/syntax_joint_ddp.sbatch configs/syntax/joint_docclean_f3_s0.json

# macronization + metrical scansion (joint), and the macron-only arm of the ablation
sbatch scripts/slurm/meter_meter.sbatch configs/meter/joint_docclean.json
sbatch scripts/slurm/meter_meter.sbatch configs/meter/mac_v2.json
```

For the leak-proof 10-fold restoration rotation used to interrogate individual
inscriptions/papyri (every document gets a fine-tuned model that provably never saw it,
regardless of which digit its ID ends in — restoration fine-tunes in about an hour, so
the full rotation is cheap):

```bash
# test_digit/val_digit rotate together: (1,2), (2,3), ..., (9,0), (0,1)
sbatch scripts/slurm/insc_finetune_whole_4node.sbatch configs/insc/finetune_whole_v4_t1v2.json
```

Each `configs/insc/finetune_whole_v*_t*v*.json` pins its own `test_digit`/
`val_digit` fields; `insc/train/finetune_whole.py` reads them and exports
`INSC_TEST_DIGIT`/`INSC_VAL_DIGIT` itself before any data loads, so the intended split
holds however the script is launched. The eval scripts
(`insc/eval/restore_strict{,_papyri}.py`) are not config-driven — when evaluating one of
these fold models, export the matching `INSC_TEST_DIGIT`/`INSC_VAL_DIGIT` yourself
before `--make-samples`/`--ckpt` (defaults to the flagship 3/4 split otherwise).

## 5. Evaluation (reproducing the paper's tables)

Every reconstruction number uses one protocol: whole documents, real lacunae left in
context as unknowns, spaces counted toward the gap length, word division predicted, beam
20, Levenshtein CER. `--make-samples` writes a frozen sample file; evaluation then reads
it, so every system sees identical gaps.

**Ten-fold rotation (Table 1).** For each held-out digit `t` (val digit `v = (t+1) % 10`),
with the arm's checkpoint from `MODEL_CARDS_INDEX.md`:

```bash
export INSC_TEST_DIGIT=3 INSC_VAL_DIGIT=4          # must match the model's fold
python -m insc.eval.restore_strict         --make-samples --n 100 --samples f3_inscr.json
python -m insc.eval.restore_strict_papyri  --make-samples --n 100 --samples f3_pap.json
python -m insc.eval.restore_strict        --ckpt <run>/best.pt --samples f3_inscr.json --out v2_d3_inscr.json
python -m insc.eval.restore_strict_papyri --ckpt <run>/best.pt --samples f3_pap.json   --out v2_d3_pap.json
```

Repeat over the ten `v4_*` runs (paper revision v2), the ten `v3_*` runs (v1), and the ten
`v3_randinit_*` runs (the matched control), then aggregate and run the fold-paired
permutation tests with `python -m analysis.sig_all`.

**Starting from a released checkpoint.** The Hub ships `model.safetensors` for
`AutoModel.from_pretrained`, while the evaluation scripts read the trainer's `.pt`. Convert once:

```bash
python scripts/hf_to_eval_checkpoint.py \
    --repo Ericu950/Stoicheia-restoration-test3 --out runs/t3v4/best.pt
```

The result is bit-identical to the checkpoint the paper evaluated (268 tensors, max absolute
difference 0.0). Export `INSC_TEST_DIGIT`/`INSC_VAL_DIGIT` to match the checkpoint's held-out
digit before running any evaluation below.

**Head-to-head against Ithaca and Aeneas (Table 2).** All three systems read the frozen
3,000-sample digit-3 file shipped in `insc/eval/frozen/`:

```bash
python -m insc.eval.restore_strict --ckpt <digit-3 run>/best.pt \
    --samples insc/eval/frozen/strict_test_fold3_samples.json --out ours_strict.json
python -m insc.eval.ithaca_baseline --samples insc/eval/frozen/strict_test_fold3_samples.json --out ithaca.json
python -m insc.eval.ptp_baseline    --ckpt <aeneas>.pkl \
    --samples insc/eval/frozen/strict_test_fold3_samples.json --out aeneas.json
```

**Recently edited documents (Tables 3-4).** Scored on the comparison release's own
documents and normalization, with every system rescored under the same Levenshtein CER:

```bash
python -m insc.eval.restore_dsh --ckpt <run>/best.pt --data <recent-inscriptions>.jsonl --out ours_recent.json
python -m insc.eval.ptp_baseline --ckpt <aeneas>.pkl --dsh <recent-inscriptions>.jsonl --out aeneas_recent.json
python -m analysis.merge_dsh          # aggregate shards, macro over gap lengths
```

**Tagging and parsing (Table 5).** The full 5-fold x 2-seed matrix per encoder:

```bash
for f in 0 1 2 3 4; do for s in 0 1; do
  sbatch scripts/slurm/syntax_joint_ddp.sbatch configs/syntax/joint_docclean_f${f}_s${s}.json
done; done
python -m parser.joint_evaluate --run $STOICHEIA_DATA/parser_data/runs/joint_docclean_f0_s0 --split test
```

**Macronization and scansion (Tables 6-7).** The macronization ablation uses the
macron-only runs (`mac_v2*`, six seeds per arm), the scansion ablation the joint runs
(`joint_docclean*` / `joint_randinit*`); the external macronizer comparison scores the
joint model on Norma's 1,916 test positions:

```bash
python -m meter.predict --model $STOICHEIA_DATA/runs/meter_joint_docclean/best.pt --norma
python -m meter.predict --model $STOICHEIA_DATA/runs/meter_mac_v2/best.pt --norma --norma-source git
```

`python -m analysis.paper_tables` collects finished evaluation logs into the table
layouts used in the paper.

## 6. Tests

```bash
pytest tests/ -q
```
CPU-only, seconds-scale. Covers normalization/packing/noising (pretraining), edit-script
lemma encoding + dataset construction (tagger), macron/scansion mark parsing (meter),
plus lightweight forward-pass smoke tests for the restoration and joint tagger/parser
pipelines (tiny randomly-initialized configs — these check the tensor plumbing survives
refactors, not model quality).

## Reproducing the 10-fold split itself

The split is not an input -- it is constructed by the 13-stage pipeline in
`data/split_pipeline/` (sentence segmentation, pristine-edition clustering via
exact/MinHash-LSH/shared-sentence evidence, zone assignment, per-fold exclusion
masks, documentary decontamination incl. blocklist, bronze back-translation
filters, verification, manifests). `data/fold_manifests/SPLIT_DESIGN.md`
documents the design; `fold_k_test_works.tsv.gz` lists every work in fold k's
test bucket (group id, kind, source, record/char counts, author+title), so the
"provably never seen" guarantee is checkable work-by-work without rebuilding
anything.

## GPU environment, baselines and ablations

**GPU environment.** `bash scripts/setup_gpu_venv.sh <venv>` installs the pins in
`requirements-gpu.txt` (Python 3.12, torch 2.12.1+cu130, aarch64 GH200). For another CUDA
version change the extra index and the torch local version.

**Baselines.** Encoders are loaded by Hub id from the configs: `bowphs/GreBerta`,
`bowphs/PhilBerta`, `pranaydeeps/Ancient-Greek-BERT`, `princeton-logion/logion-bert-base`,
`bert-base-multilingual-cased`, `xlm-roberta-base`, `xlm-roberta-large`.
Ithaca: `google-deepmind/ithaca` at `ced13193aaa52e49a2388c9ace0244e9a24e6d42`, checkpoint
`checkpoint_v1.pkl` (sha256 `d98c2a04bc2057a7c09b9edfce7b82d94931652ab858593cd12f9389f7f57afa`).
Aeneas: `google-deepmind/predictingthepast` at `a19c66aea168376ec0cab68a1fba0e378619719f`,
Greek checkpoint `ithaca_153143996_2.pkl`
(sha256 `20a21b667362f816fc77ac0b472644bcd5a22ef76e35e76b46c5b2c024b2aa54`). JAX environment: `baselines/env-jax.freeze.txt`.

**Per-sample outputs.** `restore_strict`, `ithaca_baseline` and `ptp_baseline` write a `samples`
list (`phi_id`, `L`, `start`, `gold`, `preds`, `scores`) next to the aggregates.
`scripts/appendix/parse_dump.py` and `meter.predict --norma` do the same for parsing and
macronization. The rotation samples are in `insc/eval/frozen/rotation/`.

**Ablations.** `scripts/appendix/make_ablation_configs.py` writes `configs/ablate/`;
`ablate/tok_train.py` and `ablate/tok_insc.py` train the token-level arms; `analysis/ablation_table.py`
collects the table.

## Revision naming

The paper's reconstruction revisions v1 and v2 correspond to configuration
files named `finetune_whole_v3_*` and `finetune_whole_v4_*` respectively.

## External pieces the harnesses expect

* Meter training data: `MACRONIZER_SRC` must point at a checkout of the
  `Ericu950/Stoicheia-meter-silver` dataset (silver verse lines + scanner corpus);
  the Norma benchmark itself comes from `Ericu950/norma`.
* The Ithaca baseline harness (`insc/eval/ithaca_baseline.py`) expects DeepMind's
  Ithaca repository cloned at `ithaca_upstream/` and its released checkpoint;
  the Aeneas harness (`insc/eval/ptp_baseline.py`) expects DeepMind's
  `predictingthepast` repository and its Greek checkpoint. Both are public
  third-party releases and are not vendored here.
