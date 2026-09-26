#!/usr/bin/env bash
# Compare every approach on ONE fold, without hyperparameter optimisation.
#
# Goal: sort out in a few hours which approaches deserve HPO budget. This is NOT a
# quotable result — a single fold gives no spread, and without spread no comparison
# holds (§8.4).
#
# Usage:
#   ./compare_surface.sh                 # fold 0, 100 epochs (config default)
#   ./compare_surface.sh 2 30            # fold 2, 30 epochs
#   APPROACHES="exp_a_yolo_pooled exp_d_lora" ./compare_surface.sh
#
# An approach that fails does not stop the following ones: its error is logged and the
# script goes on. That is the point of launching everything at once.

set -uo pipefail

FOLD="${1:-0}"
EPOCHS="${2:-}"
TAG="surface_f${FOLD}"
LOG_DIR="logs/surface_$(date +%Y%m%d_%H%M)"

: "${APPROACHES:=exp_a_yolo_pooled exp_b_yolo_per_dataset exp_c_detect_then_pose \
exp_d_lora exp_e_group_bn exp_f_yolo_reduced exp_g_head_only exp_h_lora_per_dataset}"

mkdir -p "$LOG_DIR"
EXTRA=()
[ -n "$EPOCHS" ] && EXTRA+=("train.epochs=$EPOCHS")

echo "=== Surface comparison | fold $FOLD | tag $TAG ==="
echo "Logs: $LOG_DIR"
[ -n "$EPOCHS" ] && echo "Forced epochs: $EPOCHS"
echo

# The split must exist: every approach shares the SAME folds (§6.2).
if ! ls data/splits/*.parquet >/dev/null 2>&1; then
    echo "No split found. Running 'split'..."
    python -m insectpose.cli split || { echo "Split FAILED, stopping."; exit 1; }
fi

declare -a SUCCEEDED=() FAILED=()
TOTAL_START=$SECONDS

for experiment in $APPROACHES; do
    echo "--- $experiment ---"
    start=$SECONDS
    if python -m insectpose.cli train \
            "experiment=$experiment" "cv.fold=$FOLD" "tag=$TAG" \
            "${EXTRA[@]}" > "$LOG_DIR/$experiment.log" 2>&1; then
        duration=$((SECONDS - start))
        SUCCEEDED+=("$experiment")
        printf '    OK   %dm%02ds\n' $((duration / 60)) $((duration % 60))
    else
        duration=$((SECONDS - start))
        FAILED+=("$experiment")
        printf '    FAILED after %dm%02ds\n' $((duration / 60)) $((duration % 60))
        echo "    Last error:"
        grep -E "Error|Exception|Traceback" "$LOG_DIR/$experiment.log" | tail -3 \
            | sed 's/^/      /'
    fi
    echo
done

TOTAL=$((SECONDS - TOTAL_START))
printf '=== %d succeeded, %d failed in %dh%02dm ===\n' \
    "${#SUCCEEDED[@]}" "${#FAILED[@]}" $((TOTAL / 3600)) $(((TOTAL % 3600) / 60))
[ "${#FAILED[@]}" -gt 0 ] && printf 'Failures: %s\n' "${FAILED[*]}"

if [ "${#SUCCEEDED[@]}" -eq 0 ]; then
    echo "No usable run."
    exit 1
fi

echo
echo "=== Aggregation ==="
python -m insectpose.cli report > "$LOG_DIR/report.log" 2>&1 \
    || { echo "Report failed, see $LOG_DIR/report.log"; exit 1; }

python - "$TAG" <<'PYEOF'
import sys
import pandas as pd
from insectpose.evaluation.aggregate import final_runs, model_label

tag = sys.argv[1]
master = final_runs(pd.read_parquet("../../results/pose/master.parquet"))
master = master[master["tag"].astype(str) == tag]
if master.empty:
    print("No run with this tag in master.parquet")
    raise SystemExit

master = master.copy()
master["model"] = model_label(master)
selection = master[(master["scope"] == "overall") & (master["split"] == "test")]

metrics = ["oks_ap", "pck@0.25_thorax_width", "kpt_coverage",
           "measurement_mape_median", "latency_ms_per_instance"]
table = selection[selection["metric"].isin(metrics)].pivot_table(
    index="model", columns="metric", values="value", aggfunc="mean")
columns = [m for m in metrics if m in table.columns]
print(table[columns].sort_values(columns[0], ascending=False).round(4).to_string())

print("\nReading reminders:")
print("  - a single fold: no spread, hence no definitive conclusion;")
print("  - read kpt_coverage BEFORE the rest: if it is low, everything is biased;")
print("  - yolo_pooled_reduced is evaluated on points it does not learn:")
print("      python scripts/compare_models.py --exclude-keypoints leg hindwing")
print("  - comparing head_only with lora tells whether the adapters bring anything.")
PYEOF

echo
echo "Details: python scripts/compare_models.py --tags $TAG"
