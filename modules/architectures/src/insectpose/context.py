"""RunContext: identity, seeds, folders and manifest of a run (§6.4, §8).

A run = a deterministic `run_id` + a folder + a manifest written LAST.
"""

from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from insectpose.contracts import MANIFEST_SCHEMA_VERSION
from insectpose.paths import ProjectPaths
from insectpose.utils.device import device_info
from insectpose.utils.hashing import short_hash, stable_hash
from insectpose.utils.io import write_json
from insectpose.utils.logging import get_logger
from insectpose.utils.seeding import seed_for, set_global_seed


def _git_state(root: Path) -> dict[str, Any]:
    """Current commit and cleanliness of the repository; 'unknown' values outside git."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": bool(status) if status is not None else None,
    }


def make_run_id(cfg: DictConfig, content_hash: str) -> str:
    """Deterministic run_id (§8.1): two identical configs give the same id.

    Format: <approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    # Purely operational keys must not change the identity of the run. `retain` is one
    # of them: exporting the model or not, and under which name, changes nothing to what
    # is trained. Leaving it in would break the idempotence (§8.1) -- `retain.name=x`
    # would retrain every fold instead of skipping them. `folds` only says which folds a
    # command runs (ADR-0039): the fold of THIS run is `fold`, which stays in the id.
    for volatile in ("force", "paths", "hydra", "retain", "folds"):
        resolved.pop(volatile, None)
    digest = short_hash(stable_hash({"cfg": resolved, "data": content_hash}))
    return "__".join(
        [
            str(cfg.approach.name),
            str(cfg.data.scope),
            str(cfg.split_id),
            f"fold{int(cfg.fold)}",
            str(cfg.tag),
            digest,
        ]
    )


def variant_hash(cfg: DictConfig, ignored_keys: list[str] | None = None) -> str:
    """Fingerprint of the MODEL, independent of the fold.

    Two runs share this fingerprint if and only if they are the same model trained on
    different folds. Without it, two variants carrying the same tag (for instance two
    different starting weights) would be averaged together in the tables: a wrong
    result, and a silent one.

    `ignored_keys` serves the nested protocol (ADR-0012), where each outer fold
    LEGITIMATELY retains different hyperparameters: these keys are excluded so that the
    folds of one experiment stay grouped.
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    for volatile in ("fold", "folds", "force", "paths", "hydra", "split_id", "retain"):
        resolved.pop(volatile, None)
    for key in ignored_keys or []:
        node = resolved
        parts = str(key).split(".")
        for part in parts[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return short_hash(stable_hash(resolved))


@dataclass
class RunContext:
    """Run context shared by every step of a run."""

    run_id: str
    cfg: DictConfig
    paths: ProjectPaths
    fold: int
    split_id: str
    content_hash: str
    started_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        """Folder of the run. No approach writes anywhere else."""
        return self.paths.run_dir(self.run_id)

    @property
    def approach_name(self) -> str:
        return str(self.cfg.approach.name)

    @property
    def logger(self) -> Any:
        return get_logger(self.run_id)

    def subdir(self, name: str) -> Path:
        """Create and return a sub-folder of the run ('weights', 'logs', 'figures'...)."""
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def seed(self, purpose: str = "global") -> int:
        """Derived seed, stable for (run_id, fold, purpose) (§6.4)."""
        return seed_for(self.run_id, self.fold, purpose, base=int(self.cfg.seed))

    def apply_seed(self, purpose: str = "global") -> int:
        """Set the python/numpy/torch RNGs and return the seed used."""
        s = self.seed(purpose)
        set_global_seed(s, deterministic=str(self.cfg.mode) == "debug")
        return s

    def setup(self) -> RunContext:
        """Create the folder of the run and write the resolved config to it BEFORE any computation."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self.apply_seed()
        return self

    def is_complete(self) -> bool:
        """True if the manifest exists: the run can be replayed and aggregated."""
        return self.paths.manifest(self.run_id).exists()

    def write_manifest(self, **fields: Any) -> Path:
        """Write `manifest.json` LAST (contract 5, §3.5).

        Side effect: writes runs/<run_id>/manifest.json.
        """
        resolved = OmegaConf.to_container(self.cfg, resolve=True)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "approach": self.approach_name,
            "data_scope": str(self.cfg.data.scope),
            "split_id": self.split_id,
            "fold": self.fold,
            "tag": str(self.cfg.tag),
            "mode": str(self.cfg.mode),
            "seed": int(self.cfg.seed),
            "content_hash": self.content_hash,
            # Identifies the MODEL, all folds together (§8.1).
            "variant_hash": self.extra.get(
                "variant_hash",
                variant_hash(self.cfg, list(self.extra.get("hpo_overridden_keys", []))),
            ),
            "eval_version": int(self.cfg.eval.version),
            "primary_metric": str(self.cfg.eval.primary_metric),
            "started_at": self.started_at,
            "finished_at": time.time(),
            "duration_s": time.time() - self.started_at,
            "git": _git_state(self.paths.root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": _package_versions(),
                # The hardware is part of the comparison conditions (ADR-0019).
                "device": device_info(self.cfg.train.get("device", "auto")),
            },
            "config": resolved,
            **self.extra,
            **fields,
        }
        return write_json(self.paths.manifest(self.run_id), manifest)


def _package_versions() -> dict[str, str]:
    """Versions of the dependencies that influence the results."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "torch", "ultralytics", "optuna", "scikit-learn", "peft"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out
