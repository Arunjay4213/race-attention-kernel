#!/usr/bin/env bash
# Entry point of the SageMaker training job that infra/sm_job.sh submits.
# The sagemaker-training toolkit extracts the source tarball into /opt/ml/code
# and runs this file from there; it then runs infra/gpu_session.sh with the
# stages in $RACE_STAGES and leaves the results where SageMaker uploads them.
#
# Exit status: nonzero only when the stage runner itself fails (bad stage name,
# missing python or torch). Test failures are recorded in the per-stage logs and
# the job still completes, so the results are uploaded either way.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DATA=/opt/ml/output/data
STAGES="${RACE_STAGES:-build sanitize test bench}"

# The runbook times and profiles one GPU. On a multi-GPU instance (p4d has 8)
# pin it to the first one so device selection never depends on defaults.
export CUDA_VISIBLE_DEVICES=0

setup_cuda() {
    # torch.utils.cpp_extension finds nvcc through CUDA_HOME; the DLC ships the
    # toolkit at /usr/local/cuda but does not always put its bin/ on PATH.
    if [ -d /usr/local/cuda/bin ]; then
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
    command -v nvcc >/dev/null || { echo "nvcc not found; cannot build the extensions" >&2; exit 1; }
}

setup_python() {
    PYTHON="$(command -v python3 || command -v python)"
    export PYTHON
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "torch sees no GPU"'
    local missing=()
    "$PYTHON" -c 'import pytest' 2>/dev/null || missing+=(pytest)
    # cpp_extension.load needs the ninja binary, which the pip package provides.
    command -v ninja >/dev/null || missing+=(ninja)
    if [ ${#missing[@]} -gt 0 ]; then
        "$PYTHON" -m pip install -q --no-cache-dir --root-user-action=ignore --disable-pip-version-check "${missing[@]}"
    fi
}

# The DLC keeps the /usr/local/cuda/bin/compute-sanitizer wrapper script but
# strips the directory it execs, so being on PATH is not enough: it has to run.
tool_works() { "$1" --version >/dev/null 2>&1; }

# Installs NVIDIA's compute-sanitizer package matching the container's nvcc
# (for example cuda-sanitizer-12-9) from NVIDIA's apt repository.
install_sanitizer() {
    local ver; ver="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')"
    local distro; distro="$(. /etc/os-release && echo "ubuntu${VERSION_ID//./}")"
    [ "$(uname -m)" = x86_64 ] || return 1
    echo "installing cuda-sanitizer-$ver from the NVIDIA $distro repository (/tmp is $(stat -c "%A %U" /tmp 2>&1))"
    # Add the repository only if absent: a second entry for the same URL with a
    # different Signed-By key makes apt-get update fail.
    if ! grep -rqs developer.download.nvidia.com/compute/cuda /etc/apt/sources.list /etc/apt/sources.list.d/; then
        local deb=/opt/ml/cuda-keyring.deb
        curl -fsSL -o "$deb" "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" &&
            dpkg -i "$deb" >/dev/null || return 1
    fi
    # apt's signature check could not create its temp file in the training
    # container's /tmp (it normally runs as the _apt user), so run apt's helpers
    # as root and give them a temp dir that is known to be writable.
    local apt_tmp=/opt/ml/apt-tmp
    mkdir -p "$apt_tmp" && chmod 1777 "$apt_tmp"
    local apt=(env TMPDIR="$apt_tmp" DEBIAN_FRONTEND=noninteractive apt-get -o APT::Sandbox::User=root -qq)
    "${apt[@]}" update &&
        "${apt[@]}" install -y --no-install-recommends "cuda-sanitizer-$ver" >/dev/null
}

# Keeps sanitize only when a working compute-sanitizer exists or can be
# installed, so the job does not spend a stage on "not found" errors.
resolve_stages() {
    local out=() s
    for s in $STAGES; do
        if [ "$s" = sanitize ] && ! tool_works compute-sanitizer; then
            if ! { install_sanitizer && tool_works compute-sanitizer; }; then
                echo "no working compute-sanitizer in this container; skipping the sanitize stage"
                continue
            fi
        fi
        out+=("$s")
    done
    # An empty list would make the runbook run every stage, so stop here instead.
    [ ${#out[@]} -gt 0 ] || { echo "no stages left to run" >&2; exit 1; }
    STAGE_LIST=("${out[@]}")
}

print_env() {
    echo "project: $PROJECT"
    echo "python: $PYTHON ($("$PYTHON" --version 2>&1))"
    echo "nvcc: $(nvcc --version | tail -1)"
    local tool
    for tool in compute-sanitizer ncu nsys ninja; do
        if tool_works "$tool"; then echo "$tool: $(command -v "$tool")"; else echo "$tool: absent or broken"; fi
    done
    echo "stages: ${STAGE_LIST[*]}"
}

main() {
    setup_cuda
    setup_python
    resolve_stages
    print_env
    # The runbook writes into $PROJECT/results. Pointing that at the output
    # directory means whatever finished is uploaded even if the job is stopped
    # or hits MaxRuntime partway through a stage.
    mkdir -p "$OUTPUT_DATA/results"
    rm -rf "$PROJECT/results"
    ln -s "$OUTPUT_DATA/results" "$PROJECT/results"
    "$PROJECT/infra/gpu_session.sh" "${STAGE_LIST[@]}"
}

main "$@"
