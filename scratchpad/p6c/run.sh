#!/bin/bash
# usage: run.sh name steps [extra args]
n=$1; st=$2; shift 2
FLYSLOP_MAX_RAM_GB=0.75 uv run --extra training python -m training.distill --overfit 100 --kind gru --steps $st --out scratchpad/p6c/$n --batch 16 --eval-every 0 --log-every 50 --gen-pairs 24 "$@" > scratchpad/p6c/$n.log 2>&1
