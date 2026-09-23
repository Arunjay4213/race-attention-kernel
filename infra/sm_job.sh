#!/usr/bin/env bash
# Run the GPU runbook (infra/gpu_session.sh) as a SageMaker training job and
# bring its results back. No SSH, no notebook: the job builds, tests and
# benchmarks, uploads results/, and the instance is released when it ends.
#
#   sm_job.sh submit [--max-runtime SECONDS] <instance-type> [stage...]
#                                   package src/ infra/ docs/, start a job, print its name
#                                   (default stages: build sanitize test bench; 2 h cap)
#   sm_job.sh status <job> [lines]  state, failure reason, billed time, last [lines=40] log lines
#   sm_job.sh wait <job>            poll until Completed/Failed/Stopped (exit 0 only on Completed)
#   sm_job.sh fetch <job>           download output.tar.gz, merge its results/ into ./results
#   sm_job.sh stop <job>            stop the job (billing stops within about two minutes)
#
# Typical use:
#   job=$(infra/sm_job.sh submit ml.g6.xlarge build test bench)
#   infra/sm_job.sh wait "$job" && infra/sm_job.sh fetch "$job"
#
# Everything for a job lives under s3://$BUCKET/jobs/<job>/ (source/ and output/).
# Results land in results/<gpu>-<timestamp>/ exactly as when the runbook runs by
# hand, plus sagemaker_job.log (the job's CloudWatch log) in each fetched run dir.
# The runbook uses one GPU; sm_entry.sh sets CUDA_VISIBLE_DEVICES=0 on multi-GPU
# instances. Override the container with RACE_SM_IMAGE=<ecr uri>.
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:-personal}" AWS_PAGER=""
REGION=us-east-1
ACCOUNT=210856421190
BUCKET="race-attention-sagemaker-$ACCOUNT"
ROLE="arn:aws:iam::$ACCOUNT:role/race-sagemaker-exec"
# AWS Deep Learning Container, PyTorch training, CUDA 12.9, Python 3.12. It ships
# nvcc and the CUDA toolkit under /usr/local/cuda, which the JIT build needs.
IMAGE="${RACE_SM_IMAGE:-763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker}"
LOG_GROUP=/aws/sagemaker/TrainingJobs
DEFAULT_STAGES=(build sanitize test bench)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Hourly on-demand SageMaker training prices in us-east-1, for the cost estimate only.
price_per_hour() {
    case "$1" in
        ml.g6.xlarge)    echo 1.2 ;;
        ml.p4d.24xlarge) echo 37.69 ;;
        *)               echo 0 ;;
    esac
}

ensure_bucket() {
    aws s3api head-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null 2>&1 && return
    # The execution role's managed policy only reaches buckets with "sagemaker" in the name.
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
    aws s3api put-public-access-block --bucket "$BUCKET" --region "$REGION" \
        --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    aws s3api put-bucket-encryption --bucket "$BUCKET" --region "$REGION" \
        --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
    echo "created s3://$BUCKET" >&2
}

describe() {
    aws sagemaker describe-training-job --region "$REGION" --training-job-name "$1" --output json
}

# Seconds the instance has been billed: the final figure once the job ends,
# otherwise the time since training started (downloading the image is billed too,
# so this slightly undercounts while the job runs).
billed_seconds() {
    python3 -c '
import json, sys
from datetime import datetime, timezone
d = json.load(sys.stdin)
if d.get("BillableTimeInSeconds"):
    print(d["BillableTimeInSeconds"])
elif d.get("TrainingStartTime"):
    # The CLI prints timestamps with the local UTC offset, which fromisoformat keeps.
    start = datetime.fromisoformat(d["TrainingStartTime"])
    print(int((datetime.now(timezone.utc) - start).total_seconds()))
else:
    print(0)' <<<"$1"
}

cmd_submit() {
    local max_runtime=7200
    if [ "${1:-}" = --max-runtime ]; then max_runtime="$2"; shift 2; fi
    local itype="${1:?usage: submit [--max-runtime SECONDS] <instance-type> [stage...]}"; shift
    local stages=("$@")
    [ ${#stages[@]} -gt 0 ] || stages=("${DEFAULT_STAGES[@]}")

    ensure_bucket
    # Names must be unique per account and at most 63 characters of [a-zA-Z0-9-].
    local name; name="race-${itype#ml.}-$(date -u +%Y%m%d-%H%M%S)"
    name="${name//./-}"
    local tarball; tarball="$(mktemp --suffix=.tar.gz)"
    # EXIT rather than RETURN: errexit leaves the function without running its RETURN trap.
    # shellcheck disable=SC2064
    trap "rm -f '$tarball'" EXIT
    tar czf "$tarball" -C "$PROJECT_DIR" --exclude=__pycache__ --exclude=results src infra docs
    local source_uri="s3://$BUCKET/jobs/$name/source/sourcedir.tar.gz"
    aws s3 cp --quiet "$tarball" "$source_uri"

    # The toolkit json-decodes every hyperparameter value, so strings are passed
    # JSON-encoded the same way the SageMaker Python SDK does.
    local spec; spec=$(jq -n \
        --arg name "$name" --arg image "$IMAGE" --arg role "$ROLE" --arg itype "$itype" \
        --arg source "$source_uri" --arg output "s3://$BUCKET/jobs/" --arg region "$REGION" \
        --arg stages "${stages[*]}" --argjson max "$max_runtime" '{
            TrainingJobName: $name,
            AlgorithmSpecification: {TrainingImage: $image, TrainingInputMode: "File"},
            RoleArn: $role,
            HyperParameters: {
                sagemaker_program: ("infra/sm_entry.sh" | tojson),
                sagemaker_submit_directory: ($source | tojson),
                sagemaker_container_log_level: "20",
                sagemaker_region: ($region | tojson)
            },
            Environment: {RACE_STAGES: $stages},
            OutputDataConfig: {S3OutputPath: $output},
            ResourceConfig: {InstanceType: $itype, InstanceCount: 1, VolumeSizeInGB: 100},
            StoppingCondition: {MaxRuntimeInSeconds: $max},
            Tags: [{Key: "project", Value: "race-attention"}]
        }')
    aws sagemaker create-training-job --region "$REGION" --cli-input-json "$spec" >/dev/null
    echo "$name"
}

cmd_status() {
    local job="${1:?usage: status <job> [lines]}" lines="${2:-40}"
    local d; d="$(describe "$job")"
    local itype secs
    itype=$(jq -r .ResourceConfig.InstanceType <<<"$d")
    secs=$(billed_seconds "$d")
    jq -r '"\(.TrainingJobName) \(.TrainingJobStatus)/\(.SecondaryStatus) \(.ResourceConfig.InstanceType)",
           (.SecondaryStatusTransitions // [] | last | "  \(.StatusMessage // "")"),
           (if .FailureReason then "  failure: \(.FailureReason)" else empty end)' <<<"$d"
    echo "  billed ${secs}s, about \$$(python3 -c "print(round($secs / 3600 * $(price_per_hour "$itype"), 2))")"
    local stream
    stream=$(aws logs describe-log-streams --region "$REGION" --log-group-name "$LOG_GROUP" \
        --log-stream-name-prefix "$job/" --output json 2>/dev/null | jq -r '.logStreams[0].logStreamName // empty')
    [ "$lines" -gt 0 ] || return 0
    if [ -z "$stream" ]; then echo "  (no log stream yet)"; return; fi
    echo "--- last $lines lines of $stream"
    aws logs get-log-events --region "$REGION" --log-group-name "$LOG_GROUP" --log-stream-name "$stream" \
        --limit "$lines" --output json | jq -r '.events[].message'
}

cmd_wait() {
    local job="${1:?usage: wait <job>}" last="" d st line failures=0
    while true; do
        # A throttled or dropped describe call must not end a multi-hour wait, but
        # an unbounded retry would spin forever on expired credentials.
        if ! d="$(describe "$job")"; then
            failures=$((failures + 1))
            [ "$failures" -lt 5 ] || { echo "describe-training-job failed 5 times in a row" >&2; return 1; }
            sleep 30; continue
        fi
        failures=0
        st=$(jq -r .TrainingJobStatus <<<"$d")
        line="$st/$(jq -r .SecondaryStatus <<<"$d")"
        [ "$line" = "$last" ] || { echo "$(date -u +%H:%M:%S) $line" >&2; last="$line"; }
        case "$st" in
            Completed) return 0 ;;
            Failed|Stopped) cmd_status "$job" 20 >&2; return 1 ;;
        esac
        sleep 30
    done
}

cmd_fetch() {
    local job="${1:?usage: fetch <job>}"
    local d; d="$(describe "$job")"
    local output; output="$(jq -r .OutputDataConfig.S3OutputPath <<<"$d" | sed 's|/*$||')/$job/output/output.tar.gz"
    local tmp; tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" EXIT
    aws s3 cp --quiet "$output" "$tmp/output.tar.gz"
    # SageMaker writes the archive with libarchive, whose extra pax headers GNU tar warns about.
    tar xzf "$tmp/output.tar.gz" -C "$tmp" --warning=no-unknown-keyword
    [ -d "$tmp/results" ] || { echo "no results/ in $output" >&2; return 1; }
    aws logs filter-log-events --region "$REGION" --log-group-name "$LOG_GROUP" \
        --log-stream-name-prefix "$job/" --output json | jq -r '.events[].message' > "$tmp/sagemaker_job.log"
    mkdir -p "$PROJECT_DIR/results"
    local run
    for run in "$tmp"/results/*/; do
        # An empty results/ (the runner never started) leaves the glob unexpanded.
        [ -d "$run" ] || { echo "no run directories in $output" >&2; return 1; }
        run="$(basename "$run")"
        cp "$tmp/sagemaker_job.log" "$tmp/results/$run/"
        # Run dirs are named by GPU and UTC timestamp, so a clash means this job was already fetched.
        if [ -e "$PROJECT_DIR/results/$run" ]; then echo "results/$run exists, left as is" >&2; continue; fi
        cp -r "$tmp/results/$run" "$PROJECT_DIR/results/"
        echo "results/$run"
    done
}

cmd_stop() {
    local job="${1:?usage: stop <job>}"
    aws sagemaker stop-training-job --region "$REGION" --training-job-name "$job"
    echo "stop requested for $job"
}

case "${1:-}" in
    submit) shift; cmd_submit "$@" ;;
    status) shift; cmd_status "$@" ;;
    wait)   shift; cmd_wait "$@" ;;
    fetch)  shift; cmd_fetch "$@" ;;
    stop)   shift; cmd_stop "$@" ;;
    *)      sed -n '2,/^set /{/^set /!p}' "$0"; exit 1 ;;
esac
