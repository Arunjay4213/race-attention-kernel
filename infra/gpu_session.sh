#!/usr/bin/env bash
# Runs the whole validation and benchmark sequence on a GPU host and collects
# every log under results/<gpu>-<timestamp>/ so the paid hour is spent running,
# not typing. Run it on the host from the synced project directory (or let
# infra/sm_entry.sh run it inside a SageMaker training job):
#
#   ./infra/gpu_session.sh [stage...]     stages: build sanitize test bench profile
#
# With no stage argument it runs them all in order and keeps going past a
# failing stage, so one broken suite does not hide the others' results.
# Python is $PYTHON if set, else python3 (or python) on PATH; it must have torch
# with CUDA. Tools that a host may lack (compute-sanitizer, ncu, nsys) are
# skipped with a message rather than recorded as failures.
# An unknown stage name exits 2 before anything runs.
set -uo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
ALL_STAGES=(build sanitize test bench profile)

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=("${ALL_STAGES[@]}")
for s in "${stages[@]}"; do
    case " ${ALL_STAGES[*]} " in
        *" $s "*) ;;
        *) echo "unknown stage '$s' (stages: ${ALL_STAGES[*]})" >&2; exit 2 ;;
    esac
done

PY="${PYTHON:-$(command -v python3 || command -v python || echo python3)}"
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
# A tool counts only if it runs: some containers keep a CUDA wrapper script
# whose real binary was stripped, so `command -v` alone would pass.
have() { "$1" --version >/dev/null 2>&1; }
skip() { echo "skipped $1: $2" | tee -a "$RESULTS/session.log"; }

stage_build() {
    run env $PY -c 'import torch, sys; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability()); print(sys.version)'
    run nvidia_smi nvidia-smi
    run build_noncausal $PY src/noncausal/build.py
    run build_causal_v2 $PY src/causal_v2/build.py
    # ptxas register/spill report lands in the build logs (-Xptxas -v); pull the summary lines out.
    # Plain grep: the GPU boxes do not have ripgrep.
    grep -hE 'Used [0-9]+ registers|spill' "$RESULTS"/build_*.log > "$RESULTS/ptxas_summary.txt" || true
}

stage_sanitize() {
    # Small cases only (infra/sanitize_cases.py): the sanitizer tools are 10-100x slower than a plain run.
    if ! have compute-sanitizer; then skip sanitize "compute-sanitizer missing or broken"; return; fi
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
    [ -f "$PROJECT/src/noncausal/bench/bench_backward.py" ] && run bench_noncausal_backward $PY src/noncausal/bench/bench_backward.py
    run bench_causal_v2 $PY src/causal_v2/bench/bench_causal.py
    # The v1 prototype is the "before" number the causal v2 speedup is quoted against.
    run bench_causal_v1 $PY src/bench_causal_v1.py
}

stage_profile() {
    # One ncu pass per extension on a mid-size problem; --set full is slow, so cap launches.
    # ncu also needs GPU performance-counter access, which containers usually deny
    # (ERR_NVGPUCTRPERM); that shows up as a nonzero exit in the ncu logs, not a skip.
    if ! have ncu; then
        skip ncu "ncu missing or broken"
    else
        run ncu_noncausal ncu --set full --launch-count 6 --export "$RESULTS/ncu_noncausal" --force-overwrite \
            $PY src/noncausal/bench/bench_forward.py --min-log2 18 --max-log2 18
        run ncu_causal ncu --set full --launch-count 6 --export "$RESULTS/ncu_causal" --force-overwrite \
            $PY src/causal_v2/bench/bench_causal.py --min-log2 18 --max-log2 18 --skip-baseline
    fi
    if ! have nsys; then skip nsys "nsys missing or broken"; return; fi
    run nsys_causal nsys profile --stats=true -o "$RESULTS/nsys_causal" --force-overwrite true \
        $PY src/causal_v2/bench/bench_causal.py --min-log2 20 --max-log2 20 --skip-baseline
}

for s in "${stages[@]}"; do "stage_$s"; done
log "done; results in $RESULTS"
