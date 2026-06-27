import optuna
import re
import numpy as np
from ultralytics import YOLO
import time
import wandb

wandb.init(project="Optuna", name=f"Study_2")

BASE_MODEL = "../yolo26s-pose.pt"
DATASET = "../datasets/Lepidoptera/yolo-config.yaml"


def attention(opt, momen, lr):
        model = YOLO(BASE_MODEL)
        model.train(data=DATASET, epochs=100, optimizer=opt, momentum=momen, lr0=lr, name="./storage/exp")
        metrics = model.val()
        return metrics.mean_results()[-1]

def objective(trial):
        opt = trial.suggest_categorical("Optimizer", ['SGD', 'Adam', 'AdamW'])
        momen = trial.suggest_float('Momentum', 0.85, 0.99)
        lr = trial.suggest_float('Learning Rate', 0.001, 0.1)
        error = attention(opt, momen, lr)
        return error

start = time.time()

study2 = optuna.create_study(direction="maximize")
study2.optimize(objective, n_trials=10)
study2.best_params

import joblib
joblib.dump(study2, "study_rfi_2.pkl")

end = time.time()
time_s = end - start
print(f'Time (s): {time_s} seconds')

# -------------- VISU

import joblib
study2 = joblib.load("study_rfi_2.pkl")

fig = optuna.visualization.plot_contour(study2)
fig.write_image("Optuna/exp2a.pdf", scale=1)

fig = optuna.visualization.plot_param_importances(study2)
fig.write_image("Optuna/exp2b.pdf", scale=1)

fig = optuna.visualization.plot_edf(study2)
fig.write_image("Optuna/exp2c.pdf", scale=1)

fig = optuna.visualization.plot_parallel_coordinate(study2)
fig.write_image("Optuna/exp2d.pdf", scale=1)

fig = optuna.visualization.plot_optimization_history(study2)
fig.write_image("Optuna/exp2e.pdf", scale=1)
