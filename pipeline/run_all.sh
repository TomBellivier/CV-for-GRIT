# export INSECTPOSE_ROOT=$(pwd)
# pip install -e ".[dev]"

# # 1. raw -> format canonique, un appel par dataset
# for d in coleoptera diptera hymenoptera lepidoptera; do
#   python -m insectpose.cli prepare data=$d
# done

# # 2. lire le rapport de couverture AVANT d'entraîner
# python -c "
# import json
# d = json.load(open('data/processed/coverage_summary.json'))
# print('absents par dataset  :', {k: len(v) for k, v in d['absent_by_dataset'].items() if v})
# print('absents PARTOUT      :', d['absent_everywhere'])
# print('présents partout     :', len(d['present_everywhere']), 'points')
# print('mesures inexploitables:', {k: len(v) for k, v in d.get('unusable_measurements_by_dataset', {}).items()})
# "

# # 3. folds externes + folds internes d'HPO, partagés par TOUTES les approches
# python -m insectpose.cli split

# # 4. un fold, pour valider le branchement
# python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 train.epochs=2

# RID=$(ls -t runs | head -1)
# ls runs/$RID                          # manifest.json présent = run complet
# ls runs/$RID/figures | head           # 12 figures pred vs GT, dont 6 pires cas
# python -c "
# import pandas as pd
# m = pd.read_parquet('runs/$RID/metrics.parquet')
# print(m[(m.scope=='overall') & (m.split=='test')][['metric','value','n']].to_string(index=False))
# "

# # 5. protocole complet : HPO nichée puis réentraînement des 5 folds externes
# python -m insectpose.cli tune experiment=exp_a_yolo_pooled

python -m insectpose.cli train experiment=exp_f_yolo_reduced cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n

python -m insectpose.cli train experiment=exp_d_lora cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n
python -m insectpose.cli train experiment=exp_e_group_bn cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n


# 6. agrégation + tableaux
python -m insectpose.cli report

