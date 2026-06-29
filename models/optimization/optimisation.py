import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend  # Optuna 4.x


import re
import numpy as np
from ultralytics import YOLO
import time
import wandb
import logging
import sys

wandb.init(project="Optuna", name=f"Study_2")

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

BASE_MODEL = "../yolo26s-pose.pt"
DATASET = "../datasets/Lepidoptera/yolo-config.yaml"

STUDY_NAME = "study_pose_kobj_box_cls_dfl_rle"

N_TRIALS = 20

SAVE = False


def attention(pose, kobj, box, cls, dfl, rle, save=True):
        model = YOLO(BASE_MODEL)
        model.train(data=DATASET, epochs=200, 
                    optimizer="SGD", lr0=0.01, pose=pose, kobj=kobj, box=box, cls=cls, dfl=dfl, rle=rle, 
                    name="./storage/exp", save=save)
        metrics = model.val()
        return metrics.mean_results()[-1]

def objective(trial):
        # opt = trial.suggest_categorical("Optimizer", ['SGD', 'Adam', 'AdamW'])
        # lr = trial.suggest_float('Learning Rate', 0.001, 0.1)
        pose = trial.suggest_float('Pose', 1, 24)
        kobj = trial.suggest_float('Keypoint Objectness', 0.5, 3)
        box = trial.suggest_float('box coeff', 1, 15)
        cls = trial.suggest_float('class coeff', 0.1, 3)
        dfl = trial.suggest_float('d. focal loss', 0.5, 3)
        rle = trial.suggest_float('rle coeff', 0.5, 3)
        error = attention(pose, kobj, box, cls, dfl, rle, save=SAVE)
        return error



if __name__ == "__main__":
        
        storage = JournalStorage(JournalFileBackend("./optuna_journal.log"))

        study2 = optuna.create_study(
                direction="maximize", study_name=STUDY_NAME, storage=storage, load_if_exists=True
                )

        start = time.time()
        study2.optimize(objective, n_trials=N_TRIALS)

        print("best parameters : ", study2.best_params)

        end = time.time()
        time_s = end - start
        print(f'time (s): {time_s} seconds')

        import joblib
        joblib.dump(study2, "study_opt_lr_pose_kobj.pkl")


# -------------- VISU

# import joblib
# study2 = joblib.load("study_rfi_2.pkl")

# fig = optuna.visualization.plot_contour(study2)
# fig.write_image("Optuna/exp2a.pdf", scale=1)

# fig = optuna.visualization.plot_param_importances(study2)
# fig.write_image("Optuna/exp2b.pdf", scale=1)

# fig = optuna.visualization.plot_edf(study2)
# fig.write_image("Optuna/exp2c.pdf", scale=1)

# fig = optuna.visualization.plot_parallel_coordinate(study2)
# fig.write_image("Optuna/exp2d.pdf", scale=1)

# fig = optuna.visualization.plot_optimization_history(study2)
# fig.write_image("Optuna/exp2e.pdf", scale=1)
