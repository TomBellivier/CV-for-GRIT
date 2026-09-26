# export INSECTPOSE_ROOT=$(pwd)
# pip install -e ".[dev]"

# create symlinks to the databases, if not already done
# ln -s /home/tombellivier/Documents/CV/CV-for-GRIT/datasets/coleoptera /home/tombellivier/Documents/CV/CV-for-GRIT/modules/architectures/data/raw
# ln -s /home/tombellivier/Documents/CV/CV-for-GRIT/datasets/diptera /home/tombellivier/Documents/CV/CV-for-GRIT/modules/architectures/data/raw
# ln -s /home/tombellivier/Documents/CV/CV-for-GRIT/datasets/hymenoptera /home/tombellivier/Documents/CV/CV-for-GRIT/modules/architectures/data/raw
# ln -s /home/tombellivier/Documents/CV/CV-for-GRIT/datasets/lepidoptera /home/tombellivier/Documents/CV/CV-for-GRIT/modules/architectures/data/raw


# # 1. raw -> canonical format, one call per dataset
# for d in coleoptera diptera hymenoptera lepidoptera; do
#   python -m insectpose.cli prepare data=$d
# done

# # 2. read the coverage report BEFORE training
# python -c "
# import json
# d = json.load(open('data/processed/coverage_summary.json'))
# print('missing per dataset    :', {k: len(v) for k, v in d['absent_by_dataset'].items() if v})
# print('missing EVERYWHERE     :', d['absent_everywhere'])
# print('present everywhere    :', len(d['present_everywhere']), 'points')
# print('unusable measurements :', {k: len(v) for k, v in d.get('unusable_measurements_by_dataset', {}).items()})
# "

# # 3. outer folds + inner HPO folds, shared by EVERY approach
# python -m insectpose.cli split

# # 4. one fold, to check the wiring
# python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0 train.epochs=2

# RID=$(ls -t runs | head -1)
# ls runs/$RID                          # manifest.json present = complete run
# ls runs/$RID/figures | head           # 12 pred vs GT figures, including 6 worst cases
# python -c "
# import pandas as pd
# m = pd.read_parquet('runs/$RID/metrics.parquet')
# print(m[(m.scope=='overall') & (m.split=='test')][['metric','value','n']].to_string(index=False))
# "

# # 5. full protocol: nested HPO then retraining of the 5 outer folds
# python -m insectpose.cli tune experiment=exp_a_yolo_pooled

## archive the runs
# mkdir -p archive/$(date +%Y%m%d)
# mv runs archive/$(date +%Y%m%d)/ && mkdir runs
# mv results archive/$(date +%Y%m%d)/ 2>/dev/null; mkdir -p results

python -m insectpose.cli train experiment=exp_f_yolo_reduced cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n

python -m insectpose.cli train experiment=exp_d_lora cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n
python -m insectpose.cli train experiment=exp_e_group_bn cv.fold=0 approach.weights=yolo26n-pose.pt tag=yolo26n


# run with logs
# python -m insectpose.cli train experiment=exp_X_XXX 2>&1 | tee logs_tune_X.txt

# 6. aggregation + tables
python -m insectpose.cli report

