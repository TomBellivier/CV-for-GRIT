#!/usr/bin/env bash
# Compare five starting weights on approach A, fold 0.
# Rough screening to settle ADR-0033: a single fold, so nothing quotable.

set -x

python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26n-pose.pt tag=base_test_yolo26n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26s-pose.pt tag=base_test_yolo26s
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo11n-pose.pt tag=base_test_yolo11n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolov8n-pose.pt tag=base_test_yolov8n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26m-pose.pt tag=base_test_yolo26m

python -m insectpose.cli report

python scripts/compare_models.py \
    --tags base_test_yolo26n base_test_yolo26s base_test_yolo26m base_test_yolo11n base_test_yolov8n \
    --out-dir ../../results/pose/comparison_base_test

set +x

# Summary table: the key metrics plus the training time, which is part of the decision
# — a gain of 2 OKS points for three times the compute is not necessarily justified,
# especially since the choice will apply to the eight approaches.
python - <<'EOF'
import pandas as pd
from insectpose.evaluation.aggregate import final_runs

m = final_runs(pd.read_parquet("../../results/pose/master.parquet"))
m = m[m.tag.astype(str).str.startswith("base_test_")].copy()
m["model"] = m.tag.str.replace("base_test_", "", regex=False)
s = m[(m.scope == "overall") & (m.split == "test")]

metrics = ["oks_ap", "pck@0.25_thorax_width", "kpt_coverage",
           "measurement_mape_median", "latency_ms_per_instance"]
t = s[s.metric.isin(metrics)].pivot_table(index="model", columns="metric", values="value")
t = t[[c for c in metrics if c in t.columns]]
t = t.join(s.groupby("model").train_time_s.mean().rename("train_s").round())
print(t.sort_values("oks_ap", ascending=False).round(4).to_string())
EOF
