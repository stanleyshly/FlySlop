#!/usr/bin/env bash
# Full offline fly-body run (PLAN §5.3 learned-task gate + §5.1 connectome experiment).
#   1. per seed: imitation from the scripted expert, then PPO fine-tuning
#   2. evaluate all seeds against random and scripted baselines on held-out tokens
#   3. the paired connectome ablation (imitation protocol, 5 seeds)
# Usage: training/run_fly_seeds.sh [extra train_ppo args, e.g. --timesteps 2000000]
# Set SCRATCH=1 to add a PPO-from-scratch arm (no imitation) per seed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/fly/sweep-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
RUN="uv run --extra training --extra connectome python -m"
for SEED in 0 1 2 3 4; do
  $RUN training.imitate --config training/ppo_fly.json --seed "$SEED" --out "$OUT/bc-seed$SEED" 2>&1 | tee "$OUT/bc-seed$SEED.log"
  $RUN training.train_ppo --config training/ppo_fly.json --seed "$SEED" --init-from "$OUT/bc-seed$SEED/bc_model.zip" \
    --name "ppo-seed$SEED" --out "$OUT" "$@" 2>&1 | tee "$OUT/ppo-seed$SEED.log"
  if [ "${SCRATCH:-0}" = "1" ]; then
    $RUN training.train_ppo --config training/ppo_fly.json --seed "$SEED" --name "scratch-seed$SEED" --out "$OUT" "$@" \
      2>&1 | tee "$OUT/scratch-seed$SEED.log"
  fi
done
$RUN training.evaluate_policy "$OUT"/ppo-seed* --episodes 100 --replay "sum;" --output "$OUT/report_ppo.json"
$RUN training.evaluate_policy "$OUT"/bc-seed* --model bc_model.zip --episodes 100 --output "$OUT/report_bc.json"
$RUN training.connectome_experiment --seeds 0 1 2 3 4 --out "$OUT/connectome"
echo "reports: $OUT/report_ppo.json $OUT/report_bc.json $OUT/connectome/report.json"
