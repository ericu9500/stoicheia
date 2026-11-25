# scripts/slurm/

These are cluster-specific SLURM `sbatch` templates, carried over from the five source repos
that were consolidated into this one (file names are prefixed by origin: `pretrain_*` from the
core pretraining repo, `tagger_*`, `syntax_*`, `meter_*`, `insc_*` from the four downstream
repos). They were written for and run on one specific SLURM cluster (GH200
nodes, apptainer containers, a particular SLURM account) and have been genericized just enough
to not contain that cluster's hardcoded account or absolute paths — they are **not**
drop-in-runnable on another cluster without editing.

Before submitting any of these, you MUST:

1. **Edit the SLURM account.** Every file has:
   ```
   #SBATCH -A YOUR_ACCOUNT   # EDIT: your SLURM account
   ```
   Replace `YOUR_ACCOUNT` with your own allocation/project account.

2. **Check the partition name.** Every file has:
   ```
   #SBATCH -p gpu   # EDIT: your GPU partition name
   ```
   `gpu` was this cluster's GPU partition name — yours may differ (`gpu-a100`, `mig`, etc).

3. **Check node/GPU counts and wall-time limits** (`--nodes`, `--gpus-per-node`, `-t`) against
   your own cluster's node shapes and queue limits — these were tuned for 4x GH200/node.

4. **Set `STOICHEIA_SIF`** (or otherwise adapt the `apptainer exec --nv $SIF ...` calls) to your
   own container image — the original NGC-based training container is not included in this
   repo. If you're not using a container at all, replace the `apptainer exec ...` wrapper with
   a plain shell invocation of the same inner command.

5. **Set `STOICHEIA_DATA`** in your login-shell environment (or export it via `--export` on
   `sbatch`) before submitting — every script assumes `$STOICHEIA_ROOT/env.sh` (sourced inside
   the job) can resolve `STOICHEIA_DATA` for shard/checkpoint/run paths.

6. **Submit from the repo root, with a `logs/` directory already there** (`mkdir -p logs`).
   Every script's `#SBATCH -o`/`-e` is a relative path (`logs/<name>-%j.out`) because
   `#SBATCH` directives are parsed statically by `sbatch` *before* the script body runs —
   a script-computed variable like `$STOICHEIA_ROOT` is never expanded there. `sbatch`
   resolves a relative `-o`/`-e` against the directory you ran `sbatch` from, which is
   reliable as long as that's the repo root and `logs/` exists first (`sbatch` does not
   create it, and job launch fails outright if it doesn't exist).

7. **`STOICHEIA_DATA` must be exported in the *submitting* shell, or passed via `--export`**
   — `env.sh` has no default for it and sourcing fails outright without one. Note that a
   job whose apptainer step runs with an empty `$SIF` will interpret the next token on the
   command line (e.g. the literal word `bash`) as the image path and fail near-instantly
   with a cryptic "could not open image .../bash" error — check `STOICHEIA_SIF` first if
   you see that.

These templates were genericized from cluster-specific originals; wall-time budgets, node
counts, and topology-specific tuning (rendezvous ports, node-local staging in
`stage_shards.sh`, etc.) were deliberately left as used and should be reviewed before
running on a different cluster.
