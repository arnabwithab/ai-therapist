#!/usr/bin/env bash
# F019: pilot gate, then automatically the full 10k pass. Idempotent: judge_v2
# resumes by (idx, instrument), so a killed run restarts where it stopped.
#
#   bash eval/run_evals_v2.sh              # pilot + full
#   PILOT_ONLY=1 bash eval/run_evals_v2.sh # stop after the gate
#
# Requires OPENCODE_GO_KEY. Logs: logs/evals-v2.log.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-data/eval/v2}
MAIN=${MAIN:-data/eval/mbench-pair-oars.csv}
PROBES=${PROBES:-data/eval/culture_probes.csv}
PROBE_RESP=${PROBE_RESP:-data/eval/culture-probe-output.csv}
JUDGES=${JUDGES:-"muse-spark-1.3-contributor deepseek-v4.1-flash"}
PILOT_N=${PILOT_N:-100}
PAIRS=${PAIRS:-300}
WORKERS=${WORKERS:-50}
mkdir -p "$OUTDIR" logs

tag() { case "$1" in *deepseek*) echo deepseek ;; *muse*) echo muse ;; *) echo "$1" ;; esac; }

# Judge input for the probes: adds the CURE reference standard per row.
if [ ! -f "$OUTDIR/probe-judge-input.csv" ]; then
  [ -f "$PROBE_RESP" ] || { echo "missing $PROBE_RESP (run the probe generation first)" >&2; exit 1; }
  uv run python - "$PROBES" "$PROBE_RESP" "$OUTDIR/probe-judge-input.csv" <<'PY'
import csv, sys
ref = {r["idx"]: r for r in csv.DictReader(open(sys.argv[1]))}
gen = list(csv.DictReader(open(sys.argv[2])))
with open(sys.argv[3], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["idx", "axis", "context", "response", "expected_standard"])
    w.writeheader()
    for r in gen:
        meta = ref.get(r["idx"], {})
        w.writerow({
            "idx": r["idx"],
            "axis": meta.get("axis", ""),
            "context": r["context"],
            "response": r["response"],
            "expected_standard": meta.get("expected_standard", ""),
        })
PY
fi

judge_rows() { # model tag n out extra...
  local model=$1 t=$2 n=$3 out=$4; shift 4
  uv run python -m eval.judge_v2 --model "$model" --workers "$WORKERS" \
    --n "$n" "$@" --out "$out" --session "ai-therapist-v2-$t"
}

if [ "${SKIP_PILOT:-0}" != "1" ]; then
echo "== pilot ($PILOT_N rows, gate criteria: completeness, no flat flag, agreement, flip, crisis regex)"
for m in $JUDGES; do
  t=$(tag "$m")
  judge_rows "$m" "$t" "$PILOT_N" "$OUTDIR/judge-$t-main.csv" \
    --baseline "$MAIN" --resp-col norag --instruments miti,safety,gdb
  judge_rows "$m" "$t" "$PILOT_N" "$OUTDIR/probes-$t.csv" \
    --baseline "$OUTDIR/probe-judge-input.csv" --resp-col response --ref-col expected_standard \
    --instruments miti,safety,culture,gdb
  judge_rows "$m" "$t" 200 "$OUTDIR/pairs-$t-pilot.csv" \
    --baseline "$MAIN" --resp-col norag --pair-col rag --both-orders
done

uv run python -m eval.evals_v2_report --gate --expected-idx "$PILOT_N" \
  --judge-a "$OUTDIR/judge-muse-main.csv" --judge-b "$OUTDIR/judge-deepseek-main.csv" \
  --pairs "$OUTDIR/pairs-muse-pilot.csv" \
  --probes "$OUTDIR/probes-muse.csv" --probe-axes "$OUTDIR/probe-judge-input.csv" \
  --json "$OUTDIR/pilot-gate.json" --markdown "$OUTDIR/pilot-gate.md"
[ "${PILOT_ONLY:-0}" = "1" ] && exit 0
fi

echo "== full 10k pass"
for m in $JUDGES; do
  t=$(tag "$m")
  judge_rows "$m" "$t" 500 "$OUTDIR/judge-$t-main.csv" --all \
    --baseline "$MAIN" --resp-col norag --instruments miti,safety,gdb
  judge_rows "$m" "$t" 100 "$OUTDIR/probes-$t.csv" --all \
    --baseline "$OUTDIR/probe-judge-input.csv" --resp-col response --ref-col expected_standard \
    --instruments miti,safety,culture,gdb
  judge_rows "$m" "$t" "$PAIRS" "$OUTDIR/pairs-$t-full.csv" --all \
    --baseline "$MAIN" --resp-col norag --pair-col rag --swap-frac 0.1
done

echo "== report"
uv run python -m eval.evals_v2_report \
  --judge-a "$OUTDIR/judge-muse-main.csv" --judge-b "$OUTDIR/judge-deepseek-main.csv" \
  --pairs "$OUTDIR/pairs-muse-full.csv" \
  --probes "$OUTDIR/probes-muse.csv" --probe-axes "$OUTDIR/probe-judge-input.csv" \
  --json "$OUTDIR/report.json" --markdown "$OUTDIR/report.md"
echo "== done: $OUTDIR/report.json"
