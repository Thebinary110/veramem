#!/usr/bin/env bash
# Full local benchmark run: guarantees (both systems) then LoCoMo.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY="${PY:-python}"
mkdir -p benchmarks/runs
$PY -m benchmarks.guarantees --systems veramem,mem0 --out benchmarks/runs/guarantees.json \
  2>&1 | tee benchmarks/runs/guarantees.log
$PY -m benchmarks.locomo --systems veramem,mem0,veramem-full --convs "${CONVS:-conv-26}" \
  2>&1 | tee benchmarks/runs/locomo.log
