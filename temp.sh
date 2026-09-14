cd models/process_folder
python process_folder.py --source folder --input "../../databases/full databases"
cd ../../pipeline/
python -m insectpose.cli tune experiment=exp_e_group_bn 2>&1 | tee logs_tune_e.txt