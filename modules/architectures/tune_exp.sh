python process_folder.py --source folder --input "../all_images/full databases"
cd ../modules/architectures/
python -m insectpose.cli tune experiment=exp_e_group_bn 2>&1 | tee logs_tune_e.txt