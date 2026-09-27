#!/usr/bin/env python3
"""
run_all.py
==========

Run the whole project, from the annotations to the analyses, as set in run_config.yaml.
The launchers (run_windows.bat, run_linux.sh, run_macos.sh) only pick the Python
interpreter and the system settings, then call this script: the same steps run in the
same way on every system.

Steps, in order (each one can be switched off in run_config.yaml):

    annotations   convert the Label Studio exports, rebuild annotation_data.csv
    images        fill the training image folders of the pose module
    check         check every annotation file (annotation_tools/check_annotations.py)
    pose          train or optimise the pose models -> retained_models/pose/
    classifiers   train the measurement-validity classifiers (optional)
    pipeline      measure the images to process, scale extraction optional
    analysis      pose report and comparison, Optuna plots, pipeline analysis

Every command and its output are also written to results/run_logs/run_<date>.log.

Usage
-----
    python run_all.py                              # settings of run_config.yaml
    python run_all.py --config my_run.yaml
    python run_all.py --dry-run                    # print the commands, run nothing
    python run_all.py --only pose analysis         # only these steps
    python run_all.py --skip annotations check     # every step but these
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent
ARCH_DIR = REPO_ROOT / "modules" / "architectures"
TRAINING_IMAGES_ROOT = ARCH_DIR / "data" / "raw"
DEFAULT_TABLE = "annotation_data/annotation_data.csv"
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))
from kp_infos import INSECT_GROUPS, KEYPOINT_NAMES  # noqa: E402

STEPS = ("annotations", "images", "check", "pose", "classifiers", "pipeline", "analysis")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Every setting with its default value: a key missing from run_config.yaml falls back
# to it, and a key unknown here is reported (typo).
DEFAULTS: dict = {
    "annotations": {
        "rebuild_table": True,
        "check": True,
        "stop_on_errors": True,
        "stop_on_warnings": False,
        "label_studio_dir": "annotation_data/label_studio_annotations",
        "pose_csv": "annotation_data/pose/pose_annotations.csv",
        "scale_csv": "annotation_data/scale/scale_annotations.csv",
        "measurements_dir": "annotation_data/meas_classifier",
        "table": DEFAULT_TABLE,
        "check_report_dir": "results/annotation_check",
    },
    "images": {
        "database_dir": "annotated_images/full databases",
        "training_images": "link",
    },
    "pose_training": {
        "enabled": True,
        "mode": "train",
        "folds": [0],
        "experiment": "exp_a_yolo_pooled",
        "epochs": 100,
        "device": "auto",
        "tag": None,
        "force": False,
        "optimisation": {
            "search": "tune_once",
            "search_fold": 0,
            "n_trials": 20,
            "inner_folds": 3,
            "final_full_fit": False,
        },
        "extra_overrides": [],
    },
    "scale_extraction": {
        "enabled": True,
        "method": "auto",
    },
    "measurement_classification": {
        "enabled": True,
        "train": True,
        "min_per_class": 20,
        "n_folds": 5,
        "compare_approaches": False,
    },
    "pipeline": {
        "enabled": True,
        "source": "folder",
        "input": "images_to_process",
        "hf_dataset": "TomBellivier/all_images",
        "hf_folders": [],
        "output": "images_to_process/results.csv",
        "models": None,
        "workers": None,
    },
    "analysis": {
        "enabled": True,
        "pose_report": True,
        "pose_comparison": True,
        "optuna_plots": True,
        "pipeline_results": True,
        "review_threshold": 0.5,
        "ground_truth": True,
        "annotated_copies": False,
        "ruler_evaluation": False,
    },
    "run": {
        "log_dir": "results/run_logs",
    },
}

CHOICES = {
    ("images", "training_images"): {"link", "copy", "none"},
    ("pose_training", "mode"): {"train", "optimise"},
    ("pose_training", "optimisation", "search"): {"tune_once", "nested"},
    ("scale_extraction", "method"): {"auto", "scale_bar", "ruler"},
    ("pipeline", "source"): {"folder", "hf"},
}


class StepFailed(Exception):
    """A step could not complete: the run stops, the reason is shown to the user."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def merge(defaults: dict, values: dict, prefix: str, unknown: list[str]) -> dict:
    out = copy.deepcopy(defaults)
    for key, value in (values or {}).items():
        name = f"{prefix}{key}"
        if key not in defaults:
            unknown.append(name)
            out[key] = value
        elif isinstance(defaults[key], dict) and isinstance(value, dict):
            out[key] = merge(defaults[key], value, f"{name}.", unknown)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> tuple[dict, list[str]]:
    if not path.is_file():
        raise StepFailed(f"Settings file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown: list[str] = []
    cfg = merge(DEFAULTS, raw, "", unknown)

    problems = []
    for keys, allowed in CHOICES.items():
        value = cfg
        for key in keys:
            value = value[key]
        if value not in allowed:
            problems.append(f"{'.'.join(keys)}: '{value}' is not one of {sorted(allowed)}")
    folds = cfg["pose_training"]["folds"]
    if not (folds == "all" or isinstance(folds, int)
            or (isinstance(folds, list) and folds and all(isinstance(f, int) for f in folds))):
        problems.append(f"pose_training.folds: {folds!r} must be a fold number, a list of "
                        "fold numbers (e.g. [0, 1, 2]) or \"all\"")
    if problems:
        raise StepFailed("Invalid settings in " + path.name + ":\n  " + "\n  ".join(problems))
    return cfg, unknown


def resolve(path_value) -> Path:
    """A path of the settings: relative to the repository root, or absolute."""
    return (REPO_ROOT / str(path_value)).resolve()


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def arg(path_value) -> str:
    """A path of the settings as a command argument: relative to the repository root
    (the commands run from there) when it is inside it, absolute otherwise."""
    return rel(resolve(path_value))


# --------------------------------------------------------------------------- #
# Command runner (console + log file)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, log_path: Path | None, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.log = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = open(log_path, "w", encoding="utf-8")
        self.env = dict(os.environ)
        # insectpose is importable even when it was not installed with pip.
        self.env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(ARCH_DIR / "src"), self.env.get("PYTHONPATH", "")) if p)
        self.env.setdefault("PYTHONUNBUFFERED", "1")
        self.env.setdefault("PYTHONIOENCODING", "utf-8")
        self.env.setdefault("MPLBACKEND", "Agg")

    def echo(self, text: str = "") -> None:
        print(text, flush=True)
        if self.log is not None:
            self.log.write(text + "\n")
            self.log.flush()

    @staticmethod
    def shown(args: list[str]) -> str:
        args = ["python" if a == PYTHON else str(a) for a in args]
        return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)

    def run(self, args: list, cwd: Path = REPO_ROOT, check: bool = True) -> int:
        args = [str(a) for a in args]
        where = "" if cwd == REPO_ROOT else f"(in {rel(cwd)}) "
        self.echo(f"$ {where}{self.shown(args)}")
        if self.dry_run:
            return 0
        process = subprocess.Popen(args, cwd=cwd, env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   errors="replace", bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            self.echo(line.rstrip("\n"))
        code = process.wait()
        if check and code != 0:
            raise StepFailed(f"Command failed (exit code {code}): {self.shown(args)}")
        return code

    def close(self) -> None:
        if self.log is not None:
            self.log.close()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def annotated_images(table_path: Path) -> dict[str, set[str]]:
    """{group: image names} of the rows pose training uses (at least one keypoint)."""
    import pandas as pd

    if not table_path.is_file():
        raise StepFailed(f"Annotation table missing: {rel(table_path)} "
                         "(enable annotations.rebuild_table).")
    table = pd.read_csv(table_path, low_memory=False)
    vis = [f"{p}_v" for p in KEYPOINT_NAMES if f"{p}_v" in table.columns]
    has_keypoint = table[vis].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
    groups: dict[str, set[str]] = {g: set() for g in INSECT_GROUPS}
    for name, group in zip(table.loc[has_keypoint, "image_name"],
                           table.loc[has_keypoint, "group"].astype(str).str.lower()):
        if group in groups:
            groups[group].add(str(name).replace("\\", "/").rsplit("/", 1)[-1])
    return groups


def training_folder(group: str) -> Path:
    return TRAINING_IMAGES_ROOT / group / "images"


def blocking_placeholder(group: str) -> Path | None:
    """raw/<group> checked out as a FILE (a symbolic link git could not create)."""
    path = TRAINING_IMAGES_ROOT / group
    return path if path.is_file() else None


def folds_override(folds) -> str:
    if folds == "all":
        return "folds=all"
    values = [folds] if isinstance(folds, int) else list(folds)
    return "folds=[" + ",".join(str(int(v)) for v in values) + "]"


def completed_pose_runs() -> int:
    runs = ARCH_DIR / "runs"
    return sum(1 for m in runs.glob("*/manifest.json")) if runs.is_dir() else 0


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def step_annotations(cfg: dict, run: Runner) -> None:
    a = cfg["annotations"]
    run.run([PYTHON, "annotation_tools/labelstudio_to_csv.py",
             "--json_files", arg(a["label_studio_dir"]), "--output", arg(a["pose_csv"])])
    run.run([PYTHON, "annotation_tools/build_annotation_data.py",
             "--pose", arg(a["pose_csv"]), "--scale", arg(a["scale_csv"]),
             "--measurements", arg(a["measurements_dir"]), "--output", arg(a["table"])])


def step_images(cfg: dict, run: Runner) -> None:
    method = cfg["images"]["training_images"]
    database = resolve(cfg["images"]["database_dir"])
    if not database.is_dir():
        run.echo(f"/!\\ Image database not found: {rel(database)}. The training folders are "
                 "left as they are.")
        return

    wanted = annotated_images(resolve(cfg["annotations"]["table"]))
    for group, names in wanted.items():
        if not names:
            continue
        placeholder = blocking_placeholder(group)
        if placeholder is not None:
            run.echo(f"/!\\ {rel(placeholder)} is a file, not a folder (a symbolic link that "
                     f"git could not create here). Delete it so that {rel(training_folder(group))} "
                     f"can be created; the '{group}' images are skipped.")
            continue
        index: dict[str, Path] = {}
        source_dir = database / group
        if source_dir.is_dir():
            for file in source_dir.rglob("*"):
                if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                    index.setdefault(file.name, file)
        target_dir = training_folder(group)
        done = linked = copied = 0
        missing: list[str] = []
        for name in sorted(names):
            target = target_dir / name
            if target.exists():
                done += 1
                continue
            source = index.get(name)
            if source is None:
                missing.append(name)
                continue
            if run.dry_run:
                linked += 1
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            if method == "link":
                try:
                    os.link(source, target)        # hard link: no extra disk space
                    linked += 1
                    continue
                except OSError:
                    pass                           # other drive or file system: copy
            shutil.copy2(source, target)
            copied += 1
        verb = "to link or copy" if run.dry_run else "linked"
        run.echo(f"{group:12} {len(names):5} annotated | {done:5} already there | "
                 f"{linked:5} {verb} | {copied:5} copied | {len(missing):5} not found in "
                 f"{rel(source_dir)}")
        if missing:
            run.echo(f"             not found, e.g. {', '.join(missing[:5])}")


def step_check(cfg: dict, run: Runner) -> None:
    a = cfg["annotations"]
    args = [PYTHON, "annotation_tools/check_annotations.py",
            "--label-studio", arg(a["label_studio_dir"]),
            "--pose", arg(a["pose_csv"]), "--scale", arg(a["scale_csv"]),
            "--measurements", arg(a["measurements_dir"]),
            "--annotation-table", arg(a["table"]),
            "--images-dir", arg(cfg["images"]["database_dir"]),
            "--min-per-class", cfg["measurement_classification"]["min_per_class"],
            "--report-dir", arg(a["check_report_dir"])]
    if a["stop_on_warnings"]:
        args.append("--strict")
    if run.run(args, check=False) == 0:
        return
    if a["stop_on_errors"] or a["stop_on_warnings"]:
        raise StepFailed("The annotation check found problems (see above, and "
                         "results/annotation_check/issues.csv). Fix them, or set "
                         "annotations.stop_on_errors: false to go on anyway.")
    run.echo("/!\\ The annotation check found problems; going on (annotations.stop_on_errors: false).")


def pose_overrides(cfg: dict) -> list[str]:
    """Hydra overrides shared by every command of the pose module."""
    p = cfg["pose_training"]
    overrides = [f"experiment={p['experiment']}"]
    table = resolve(cfg["annotations"]["table"])
    if table != resolve(DEFAULT_TABLE):
        # Only when it differs: the value enters the run_id of the pose runs.
        overrides.append(f"data.adapter_options.csv_path='{table.as_posix()}'")
    return overrides


def check_training_images(cfg: dict, run: Runner) -> None:
    """Fail before training rather than in the middle of it: the YOLO export stops at the
    first missing image."""
    wanted = annotated_images(resolve(cfg["annotations"]["table"]))
    missing = {g: [n for n in names if not (training_folder(g) / n).is_file()]
               for g, names in wanted.items()}
    total = sum(len(v) for v in missing.values())
    if total == 0:
        return
    lines = [f"  {g}: {len(v)} of {len(wanted[g])} missing, e.g. {', '.join(sorted(v)[:3])}"
             for g, v in missing.items() if v]
    message = (f"{total} annotated image(s) missing from {rel(TRAINING_IMAGES_ROOT)}/<group>/images/:\n"
               + "\n".join(lines)
               + "\nPut the images in images.database_dir (one folder per group) with "
                 "images.training_images: link, or fill these folders yourself.")
    if run.dry_run:
        run.echo("/!\\ (dry run) " + message)
        return
    raise StepFailed(message)


def step_pose(cfg: dict, run: Runner) -> None:
    p = cfg["pose_training"]
    cli = [PYTHON, "-m", "insectpose.cli"]
    common = pose_overrides(cfg)
    run.run([*cli, "prepare", *common], cwd=ARCH_DIR)
    run.run([*cli, "split", *common], cwd=ARCH_DIR)
    check_training_images(cfg, run)

    overrides = [*common, folds_override(p["folds"]),
                 f"train.epochs={int(p['epochs'])}", f"train.device='{p['device']}'"]
    if p["tag"]:
        overrides.append(f"tag='{p['tag']}'")
    if p["force"]:
        overrides.append("force=true")
    if p["mode"] == "optimise":
        o = p["optimisation"]
        overrides += [f"tuning.mode={o['search']}", f"tuning.n_trials={int(o['n_trials'])}",
                      f"tuning.inner_folds={int(o['inner_folds'])}",
                      f"tuning.tuning_outer_fold={int(o['search_fold'])}",
                      f"tuning.final_full_fit={str(bool(o['final_full_fit'])).lower()}"]
    overrides += [str(o) for o in p["extra_overrides"]]
    verb = "train" if p["mode"] == "train" else "tune"
    run.run([*cli, verb, *overrides], cwd=ARCH_DIR)


def step_classifiers(cfg: dict, run: Runner) -> None:
    m = cfg["measurement_classification"]
    run.run([PYTHON, "modules/meas_classifier/train_measure_validity.py",
             "--annotation-data", arg(cfg["annotations"]["table"]),
             "--min-per-class", int(m["min_per_class"]), "--n-folds", int(m["n_folds"])])
    if m["compare_approaches"]:
        run.run([PYTHON, "modules/meas_classifier/compare_measure_validity_approaches.py"])


def step_pipeline(cfg: dict, run: Runner) -> None:
    p = cfg["pipeline"]
    s = cfg["scale_extraction"]
    scale = s["method"] if s["enabled"] else "none"
    scale_bar_model = REPO_ROOT / "retained_models" / "scale_bar" / "best.pt"
    if scale in ("auto", "scale_bar") and not scale_bar_model.is_file() and not run.dry_run:
        raise StepFailed(f"Scale-bar detector missing: {rel(scale_bar_model)}. Put it there, or "
                         "set scale_extraction.method: ruler (or enabled: false).")

    args = [PYTHON, "pipeline/process_folder.py", "--source", p["source"]]
    if p["source"] == "folder":
        args += ["--input", arg(p["input"])]
    else:
        args += ["--dataset", p["hf_dataset"]]
        if p["hf_folders"]:
            args += ["--hf-folders", *[str(f) for f in p["hf_folders"]]]
    args += ["--output", arg(p["output"]), "--scale", scale]
    if not cfg["measurement_classification"]["enabled"]:
        args.append("--no-measurement-classifier")
    if p["models"]:
        args += ["--models", arg(p["models"])]
    if p["workers"]:
        args += ["--workers", int(p["workers"])]
    run.run(args)


def step_analysis(cfg: dict, run: Runner) -> list[str]:
    """Every analysis runs even if another one fails: the failures are returned."""
    a = cfg["analysis"]
    failed: list[str] = []

    def attempt(label: str, args: list, cwd: Path = REPO_ROOT) -> bool:
        if run.run(args, cwd=cwd, check=False) != 0:
            failed.append(label)
            run.echo(f"/!\\ {label} failed; the other analyses go on.")
            return False
        return True

    cli = [PYTHON, "-m", "insectpose.cli"]
    has_runs = run.dry_run or completed_pose_runs() > 0
    if a["pose_report"] and has_runs:
        if attempt("pose report", [*cli, "report"], ARCH_DIR) and a["pose_comparison"]:
            attempt("pose comparison", [PYTHON, "scripts/compare_models.py"], ARCH_DIR)
    elif a["pose_report"]:
        run.echo("No complete pose run in modules/architectures/runs/: pose report skipped.")

    optimised = cfg["pose_training"]["enabled"] and cfg["pose_training"]["mode"] == "optimise"
    if a["optuna_plots"] and (optimised or (ARCH_DIR / "runs" / "optuna").is_dir()):
        attempt("Optuna plots", [PYTHON, "plot_optuna.py"], ARCH_DIR)

    output = resolve(cfg["pipeline"]["output"])
    if a["pipeline_results"] and (run.dry_run or output.is_file()):
        args = [PYTHON, "pipeline/analyze_results.py", "--input", arg(cfg["pipeline"]["output"]),
                "--review-threshold", float(a["review_threshold"])]
        if not a["ground_truth"]:
            args.append("--no-gt")
        if a["annotated_copies"]:
            if cfg["pipeline"]["source"] == "folder":
                args += ["--print", "--images-dir", arg(cfg["pipeline"]["input"])]
            else:
                run.echo("analysis.annotated_copies needs a local image folder: skipped.")
        attempt("pipeline analysis", args)
    elif a["pipeline_results"]:
        run.echo(f"No pipeline output at {rel(output)}: pipeline analysis skipped.")

    if a["ruler_evaluation"]:
        attempt("ruler evaluation", [PYTHON, "modules/ruler_detection/ruler_detection_evaluation.py"])
    return failed


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def plan_lines(cfg: dict) -> list[str]:
    p, s, m = cfg["pose_training"], cfg["scale_extraction"], cfg["measurement_classification"]
    pose = "off"
    if p["enabled"]:
        pose = f"{p['mode']} {p['experiment']}, folds {p['folds']}, {p['epochs']} epochs, device {p['device']}"
        if p["mode"] == "optimise":
            o = p["optimisation"]
            pose += f", {o['search']} search of {o['n_trials']} trials"
    source = cfg["pipeline"]["input"] if cfg["pipeline"]["source"] == "folder" \
        else f"hf:{cfg['pipeline']['hf_dataset']}"
    return [
        f"  annotation table     {'rebuilt' if cfg['annotations']['rebuild_table'] else 'as it is'}"
        f", check {'on' if cfg['annotations']['check'] else 'off'}",
        f"  pose training        {pose}",
        f"  scale extraction     {s['method'] if s['enabled'] else 'off'}",
        f"  measurement valid.   {('on, retrained' if m['train'] else 'on') if m['enabled'] else 'off'}",
        f"  pipeline             {source if cfg['pipeline']['enabled'] else 'off'}"
        f" -> {cfg['pipeline']['output']}",
        f"  analyses             {'on' if cfg['analysis']['enabled'] else 'off'}",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the whole project as set in run_config.yaml.")
    p.add_argument("--config", default="run_config.yaml", help="Settings file (default: run_config.yaml).")
    p.add_argument("--dry-run", action="store_true", help="Print the commands, run nothing.")
    p.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps.")
    p.add_argument("--skip", nargs="+", choices=STEPS, default=[], help="Skip these steps.")
    return p.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")   # a console that cannot show a character
        except AttributeError:
            pass
    args = parse_args()
    config_path = resolve(args.config)
    try:
        cfg, unknown = load_config(config_path)
    except StepFailed as exc:
        print(f"ERROR: {exc}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None if args.dry_run else resolve(cfg["run"]["log_dir"]) / f"run_{stamp}.log"
    run = Runner(log_path, args.dry_run)
    selected = [s for s in STEPS if (not args.only or s in args.only) and s not in args.skip]
    pose_on = bool(cfg["pose_training"]["enabled"])
    classification = cfg["measurement_classification"]
    enabled = {   # what run_config.yaml switches on
        "annotations": bool(cfg["annotations"]["rebuild_table"]),
        "images": pose_on and cfg["images"]["training_images"] != "none",
        "check": bool(cfg["annotations"]["check"]),
        "pose": pose_on,
        "classifiers": bool(classification["enabled"] and classification["train"]),
        "pipeline": bool(cfg["pipeline"]["enabled"]),
        "analysis": bool(cfg["analysis"]["enabled"]),
    }
    functions = {"annotations": step_annotations, "images": step_images, "check": step_check,
                 "pose": step_pose, "classifiers": step_classifiers,
                 "pipeline": step_pipeline, "analysis": step_analysis}

    run.echo("=" * 78)
    run.echo(f"CV-for-GRIT run {stamp}{'  (dry run: nothing is executed)' if args.dry_run else ''}")
    run.echo(f"Settings: {rel(config_path)}   Python: {PYTHON}")
    run.echo("=" * 78)
    for line in plan_lines(cfg):
        run.echo(line)
    for key in unknown:
        run.echo(f"/!\\ Unknown setting '{key}' in {config_path.name} (typo?): ignored.")
    if log_path is not None:
        run.echo(f"Log: {rel(log_path)}")
        run.log.write("\n--- settings ---\n" + yaml.safe_dump(cfg, sort_keys=False) + "---\n")

    results: list[tuple[str, str, float]] = []
    failure = None
    for step in STEPS:
        if step not in selected:
            results.append((step, "skipped (--only / --skip)", 0.0))
            continue
        if not enabled[step]:
            results.append((step, "skipped (off in the settings)", 0.0))
            continue
        run.echo("")
        run.echo(f"===== {step} " + "=" * (70 - len(step)))
        start = time.time()
        try:
            outcome = functions[step](cfg, run)
            status = "dry run" if args.dry_run else "done"
            if step == "analysis" and outcome:
                status = f"partial ({', '.join(outcome)} failed)"
                failure = failure or "analysis"
            results.append((step, status, time.time() - start))
        except StepFailed as exc:
            results.append((step, "FAILED", time.time() - start))
            run.echo(f"\nERROR in step '{step}': {exc}")
            failure = step
            break

    run.echo("")
    run.echo("=" * 78)
    run.echo("Summary")
    for step, status, seconds in results:
        duration = f"{seconds / 60:6.1f} min" if seconds >= 1 else ""
        run.echo(f"  {step:12} {status:40} {duration}")
    if not args.dry_run:
        pose_dir = REPO_ROOT / "retained_models" / "pose"
        n_models = len(list(pose_dir.rglob("*.pt"))) if pose_dir.is_dir() else 0
        run.echo(f"  pose ensemble: {n_models} model(s) in {rel(pose_dir)}")
        output = resolve(cfg["pipeline"]["output"])
        if output.is_file():
            run.echo(f"  pipeline output: {rel(output)}")
        run.echo(f"  analyses: {rel(REPO_ROOT / 'results')}/   log: {rel(log_path)}")
    run.close()
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
