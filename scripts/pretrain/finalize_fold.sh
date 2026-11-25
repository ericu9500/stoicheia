#!/bin/bash
# After a fold's chain prints DONE: report the model of record and optionally prune the
# redundant checkpoints (last.pt duplicates final.pt at completion; pre_anneal.pt only
# matters for anneal re-runs). Never deletes final.pt / best.pt / eval.jsonl.
#
# Usage: scripts/pretrain/finalize_fold.sh <fold 0-9> [--prune]
set -euo pipefail
FOLD=${1:?usage: finalize_fold.sh <fold 0-9> [--prune]}
STOICHEIA_DATA=${STOICHEIA_DATA:?set STOICHEIA_DATA/STOICHEIA_DATA (source env.sh first)}
STOICHEIA_ROOT=${STOICHEIA_ROOT:?set STOICHEIA_ROOT/STOICHEIA_ROOT (source env.sh first)}
RUN=$STOICHEIA_DATA/runs/stoicheia_fold_$FOLD

LOG=$(grep -l "DONE (end step" "$STOICHEIA_ROOT"/logs/fold${FOLD}-*.out 2>/dev/null | tail -1 || true)
if [ -z "$LOG" ]; then
  echo "fold $FOLD: no DONE marker in logs yet — not finalizing"; exit 1
fi
[ -f "$RUN/final.pt" ] || { echo "fold $FOLD: DONE logged but $RUN/final.pt missing"; exit 1; }

if grep -q "EARLY-STOPPED" "$LOG"; then
  echo "fold $FOLD: COMPLETE (early-stopped) — model of record: $RUN/best.pt"
else
  echo "fold $FOLD: COMPLETE — model of record: $RUN/final.pt"
fi
tail -3 "$RUN/eval.jsonl" 2>/dev/null || true

if [ "${2:-}" = "--prune" ]; then
  rm -fv "$RUN/last.pt" "$RUN/pre_anneal.pt"
fi
du -sh "$RUN"
