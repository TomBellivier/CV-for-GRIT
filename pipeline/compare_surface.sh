#!/usr/bin/env bash
# Compare toutes les approches sur UN fold, sans optimisation d'hyperparametres.
#
# Objectif : degrossir en quelques heures quelles approches meritent qu'on depense du
# budget d'HPO dessus. Ce n'est PAS un resultat citable — un seul fold ne donne aucune
# dispersion, et sans dispersion aucune comparaison ne tient (§8.3).
#
# Usage :
#   ./compare_surface.sh                 # fold 0, 100 epoques (defaut des configs)
#   ./compare_surface.sh 2 30            # fold 2, 30 epoques
#   APPROACHES="exp_a_yolo_pooled exp_d_lora" ./compare_surface.sh
#
# Une approche qui echoue n'interrompt pas les suivantes : son erreur est journalisee
# et le script continue. C'est le point de tout lancer d'un coup.

set -uo pipefail

FOLD="${1:-0}"
EPOCHS="${2:-}"
TAG="surface_f${FOLD}"
LOG_DIR="logs/surface_$(date +%Y%m%d_%H%M)"

: "${APPROACHES:=exp_a_yolo_pooled exp_b_yolo_per_dataset exp_c_detect_then_pose \
exp_d_lora exp_e_group_bn exp_f_yolo_reduced exp_g_head_only}"

mkdir -p "$LOG_DIR"
EXTRA=()
[ -n "$EPOCHS" ] && EXTRA+=("train.epochs=$EPOCHS")

echo "=== Comparaison de surface | fold $FOLD | tag $TAG ==="
echo "Journaux : $LOG_DIR"
[ -n "$EPOCHS" ] && echo "Epoques forcees : $EPOCHS"
echo

# Le decoupage doit exister : toutes les approches partagent les MEMES folds (§6.2).
if ! ls data/splits/*.parquet >/dev/null 2>&1; then
    echo "Aucun decoupage trouve. Lancement de 'split'..."
    python -m insectpose.cli split || { echo "ECHEC du decoupage, arret."; exit 1; }
fi

declare -a REUSSIES=() ECHOUEES=()
DEBUT_TOTAL=$SECONDS

for experiment in $APPROACHES; do
    echo "--- $experiment ---"
    debut=$SECONDS
    if python -m insectpose.cli train \
            "experiment=$experiment" "cv.fold=$FOLD" "tag=$TAG" \
            "${EXTRA[@]}" > "$LOG_DIR/$experiment.log" 2>&1; then
        duree=$((SECONDS - debut))
        REUSSIES+=("$experiment")
        printf '    OK   %dm%02ds\n' $((duree / 60)) $((duree % 60))
    else
        duree=$((SECONDS - debut))
        ECHOUEES+=("$experiment")
        printf '    ECHEC apres %dm%02ds\n' $((duree / 60)) $((duree % 60))
        echo "    Derniere erreur :"
        grep -E "Error|Exception|Traceback" "$LOG_DIR/$experiment.log" | tail -3 \
            | sed 's/^/      /'
    fi
    echo
done

TOTAL=$((SECONDS - DEBUT_TOTAL))
printf '=== %d reussie(s), %d echouee(s) en %dh%02dm ===\n' \
    "${#REUSSIES[@]}" "${#ECHOUEES[@]}" $((TOTAL / 3600)) $(((TOTAL % 3600) / 60))
[ "${#ECHOUEES[@]}" -gt 0 ] && printf 'Echecs : %s\n' "${ECHOUEES[*]}"

if [ "${#REUSSIES[@]}" -eq 0 ]; then
    echo "Aucun run exploitable."
    exit 1
fi

echo
echo "=== Agregation ==="
python -m insectpose.cli report > "$LOG_DIR/report.log" 2>&1 \
    || { echo "Echec du rapport, voir $LOG_DIR/report.log"; exit 1; }

python - "$TAG" <<'PYEOF'
import sys
import pandas as pd
from insectpose.evaluation.aggregate import final_runs, model_label

tag = sys.argv[1]
master = final_runs(pd.read_parquet("results/master.parquet"))
master = master[master["tag"].astype(str) == tag]
if master.empty:
    print("Aucun run avec ce tag dans master.parquet")
    raise SystemExit

master = master.copy()
master["model"] = model_label(master)
selection = master[(master["scope"] == "overall") & (master["split"] == "test")]

metriques = ["oks_ap", "pck@0.25_thorax_width", "kpt_coverage",
             "measurement_mape_median", "latency_ms_per_instance"]
table = selection[selection["metric"].isin(metriques)].pivot_table(
    index="model", columns="metric", values="value", aggfunc="mean")
colonnes = [m for m in metriques if m in table.columns]
print(table[colonnes].sort_values(colonnes[0], ascending=False).round(4).to_string())

print("\nRappels de lecture :")
print("  - un seul fold : aucune dispersion, donc aucune conclusion definitive ;")
print("  - lire kpt_coverage AVANT le reste : si elle est basse, tout est biaise ;")
print("  - yolo_pooled_reduced est evaluee sur des points qu'elle n'apprend pas :")
print("      python scripts/compare_models.py --exclude-keypoints leg hindwing")
print("  - comparer head_only a lora dit si les adaptateurs apportent quelque chose.")
PYEOF

echo
echo "Detail : python scripts/compare_models.py --tags $TAG"