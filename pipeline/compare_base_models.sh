#!/usr/bin/env bash
# Compare cinq poids de depart sur l'approche A, fold 0.
# Degrossissage pour trancher ADR-0033 : un seul fold, donc rien de citable.

set -x

python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26n-pose.pt tag=base_test_yolo26n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26s-pose.pt tag=base_test_yolo26s
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo11n-pose.pt tag=base_test_yolo11n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolov8n-pose.pt tag=base_test_yolov8n
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 approach.weights=yolo26m-pose.pt tag=base_test_yolo26m

python -m insectpose.cli report

python scripts/compare_models.py \
    --tags base_test_yolo26n base_test_yolo26s base_test_yolo26m base_test_yolo11n base_test_yolov8n \
    --out-dir results/comparison_base_test

set +x

# Tableau recapitulatif : les metriques cles plus le temps d'entrainement, qui fait
# partie de la decision — un gain de 2 points d'OKS pour trois fois le calcul ne se
# justifie pas forcement, d'autant que le choix vaudra pour les huit approches.
python - <<'EOF'
import pandas as pd
from insectpose.evaluation.aggregate import final_runs

m = final_runs(pd.read_parquet("results/master.parquet"))
m = m[m.tag.astype(str).str.startswith("base_test_")].copy()
m["model"] = m.tag.str.replace("base_test_", "", regex=False)
s = m[(m.scope == "overall") & (m.split == "test")]

metriques = ["oks_ap", "pck@0.25_thorax_width", "kpt_coverage",
             "measurement_mape_median", "latency_ms_per_instance"]
t = s[s.metric.isin(metriques)].pivot_table(index="model", columns="metric", values="value")
t = t[[c for c in metriques if c in t.columns]]
t = t.join(s.groupby("model").train_time_s.mean().rename("train_s").round())
print(t.sort_values("oks_ap", ascending=False).round(4).to_string())
EOF