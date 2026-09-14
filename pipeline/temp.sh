./compare_surface.sh

python -m insectpose.cli tune experiment=exp_h_lora_per_dataset 2>&1 | tee logs_tune_h.txt
