cd lora_adapters

python train_lora.py \
    --base-weights best.pt \
    --data-config groups.yaml \
    --rank 8 --targets neck_head \
    --epochs 80 --batch 16 --lr0 0.001 \
    --out-dir lora_weights --runs-dir runs_lora

python eval_lora.py \
    --manifest lora_weights/lora_manifest.json \
    --out-dir pose_results

cd ../group_batchnorm

python train_group_bn.py \
    --base-weights best.pt \
    --data-config groups.yaml \
    --epochs 40 --lr0 0.002 \
    --out-dir gbn_weights --runs-dir runs_gbn \
    --verify-shared

python eval_group_bn.py \
    --manifest gbn_weights/gbn_manifest.json \
    --out-dir pose_results