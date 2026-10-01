#!/usr/bin/env bash
# Full offline run: train 5 seeds, then evaluate them together against baselines.
# Usage: training/run_seeds.sh [extra train_ppo args...]   e.g. --timesteps 5000000 --subproc
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=runs/ppo/sweep-$STAMP
mkdir -p "$OUT"
for SEED in 0 1 2 3 4; do
  uv run --extra training python -m training.train_ppo --seed "$SEED" --name "seed$SEED" --out "$OUT" "$@" \
    2>&1 | tee "$OUT/seed$SEED.log"
done
uv run --extra training python -m training.evaluate_policy "$OUT"/seed* --episodes 100 \
  --replay "sum;" --output "$OUT/report.json"
echo "report: $OUT/report.json"
