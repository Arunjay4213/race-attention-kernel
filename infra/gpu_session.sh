#!/usr/bin/env bash
# Runs the whole validation and benchmark sequence on the GPU box and collects
# every log under results/<gpu>-<timestamp>/ so the paid hour is spent running,
# not typing. Run it on the box from the synced project directory:
#
#   ./infra/gpu_session.sh [stage...]     stages: build sanitize test bench profile
#
# With no stage argument it runs them all in order and keeps going past a
# failing stage, so one broken suite does not hide the others' results.
# Python is the DLAMI's torch environment (python3 on PATH after `source
# /opt/pytorch/bin/activate` or equivalent; see DLAMI release notes).
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
GPU_NAME="$($PY -c 'import torch; print(torch.cuda.get_device_name().replace(" ", "_"))' 2>/dev/null || echo unknown)"
RESULTS="$PROJECT/results/${GPU_NAME}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RESULTS"
export TORCH_EXTENSIONS_DIR="$HOME/.cache/race_torch_ext"

log() { printf '\n=== %s (%s) ===\n' "$1" "$(date -u +%H:%M:%S)" | tee -a "$RESULTS/session.log"; }
run() {
    # run <name> <cmd...>: tee output to its own log, record the exit code, never abort the session
    local name="$1"; shift
    log "$name"
    ( cd "$PROJECT" && "$@" ) 2>&1 | tee "$RESULTS/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$name exit=$rc" | tee -a "$RESULTS/session.log"
}

stage_build() {
    run env $PY -c 'import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); print(sys.version)'
    run nvidia_smi nvidia-smi
    run build_noncausal $PY src/noncausal/build.py
    run build_causal_v2 $PY src/causal_v2/build.py
    # ptxas register/spill report lands in the build logs (-Xptxas -v); pull the summary lines out
    rg --color=never -N 'Used [0-9]+ registers|spill' "$RESULTS"/build_*.log > "$RESULTS/ptxas_summary.txt" || true
}

stage_sanitize() {
    # Small cases only (infra/sanitize_cases.py): the sanitizer tools are 10-100x slower than a plain run.
    for tool in memcheck racecheck synccheck; do
        run "sanitize_$tool" compute-sanitizer --tool "$tool" --error-exitcode 1 $PY infra/sanitize_cases.py
    done
}

stage_test() {
    run test_noncausal $PY -m pytest src/noncausal/tests -q -p no:cacheprovider
    run test_causal_v2 $PY -m pytest src/causal_v2/tests -q -p no:cacheprovider
}

stage_bench() {
    run bench_noncausal_forward $PY src/noncausal/bench/bench_forward.py
    [ -f src/noncausal/bench/bench_backward.py ] && run bench_noncausal_backward $PY src/noncausal/bench/bench_backward.py
    run bench_causal_v2 $PY src/causal_v2/bench/bench_causal.py
    run peak_memory $PY -c 'import torch; print("peak allocated GB", torch.cuda.max_memory_allocated() / 2**30)'
}

stage_profile() {
    # One ncu pass per extension on a mid-size problem; --set full is slow, so cap launches.
    run ncu_noncausal ncu --set full --launch-count 6 --export "$RESULTS/ncu_noncausal" --force-overwrite \
        $PY src/noncausal/bench/bench_forward.py --min-log2 18 --max-log2 18
    run ncu_causal ncu --set full --launch-count 6 --export "$RESULTS/ncu_causal" --force-overwrite \
        $PY src/causal_v2/bench/bench_causal.py --min-log2 18 --max-log2 18 --skip-baseline
    run nsys_causal nsys profile --stats=true -o "$RESULTS/nsys_causal" --force-overwrite true \
        $PY src/causal_v2/bench/bench_causal.py --min-log2 20 --max-log2 20 --skip-baseline
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(build sanitize test bench profile)
for s in "${stages[@]}"; do "stage_$s"; done
log "done; results in $RESULTS"
