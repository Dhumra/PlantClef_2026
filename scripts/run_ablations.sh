#!/bin/bash
# Run ablation submissions in parallel (all cache-based, no GPU needed).
# Logs go to logs/ablation_<name>.log
# Usage: bash scripts/run_ablations.sh

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p output logs

echo "==== Ablation runs started: $(date) ===="

python -m pipeline.run_pipeline \
    --scales 6 5 4 3 2 1 --aggregation max \
    --no-bayesian-prior --no-geo-filter \
    --output output/paper1_repro.csv \
    > logs/ablation_paper1_repro.log 2>&1 &
PID_P1=$!

python -m pipeline.run_pipeline \
    --scales 4 --aggregation mean \
    --output output/paper2_repro.csv \
    > logs/ablation_paper2_repro.log 2>&1 &
PID_P2=$!

python -m pipeline.run_pipeline \
    --scales 6 5 4 3 2 1 --aggregation max \
    --output output/submission_combined.csv \
    > logs/ablation_combined.log 2>&1 &
PID_COMB=$!

echo "Launched:"
echo "  paper1_repro    (PID $PID_P1)  -> logs/ablation_paper1_repro.log"
echo "  paper2_repro    (PID $PID_P2)  -> logs/ablation_paper2_repro.log"
echo "  combined        (PID $PID_COMB) -> logs/ablation_combined.log"
echo ""
echo "Waiting for all to finish..."

wait $PID_P1  && echo "  [DONE] paper1_repro"    || echo "  [FAIL] paper1_repro - check logs/ablation_paper1_repro.log"
wait $PID_P2  && echo "  [DONE] paper2_repro"    || echo "  [FAIL] paper2_repro - check logs/ablation_paper2_repro.log"
wait $PID_COMB && echo "  [DONE] combined"        || echo "  [FAIL] combined - check logs/ablation_combined.log"

echo ""
echo "==== All done: $(date) ===="
echo "Output files:"
ls -lh output/*.csv 2>/dev/null || echo "  (none found)"
