"""RunContext : identite, seeds, repertoires et manifeste d'une execution (§6.4, §8).

Un run = un `run_id` deterministe + un repertoire + un manifeste ecrit EN DERNIER.
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
    """Commit courant et proprete du depot ; valeurs 'unknown' hors depot git."""

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
    """run_id deterministe (§8.1) : deux configs identiques donnent le meme id.

    Format : <approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
    """
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    # Les cles purement operationnelles ne doivent pas changer l'identite du run.
    for volatile in ("force", "paths", "hydra"):
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


@dataclass
class RunContext:
    """Contexte d'execution partage par toutes les etapes d'un run."""

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
        """Repertoire du run. Aucune approche n'ecrit ailleurs."""
        return self.paths.run_dir(self.run_id)

    @property
    def approach_name(self) -> str:
        return str(self.cfg.approach.name)

    @property
    def logger(self) -> Any:
        return get_logger(self.run_id)

    def subdir(self, name: str) -> Path:
        """Cree et retourne un sous-repertoire du run ('weights', 'logs', 'figures'...)."""
        d = self.run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def seed(self, purpose: str = "global") -> int:
        """Seed derivee, stable pour (run_id, fold, purpose) (§6.4)."""
        return seed_for(self.run_id, self.fold, purpose, base=int(self.cfg.seed))

    def apply_seed(self, purpose: str = "global") -> int:
        """Fixe les RNG python/numpy/torch et retourne la seed utilisee."""
        s = self.seed(purpose)
        set_global_seed(s, deterministic=str(self.cfg.mode) == "debug")
        return s

    def setup(self) -> RunContext:
        """Cree le repertoire du run et y ecrit la config resolue AVANT tout calcul."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.yaml").write_text(
            OmegaConf.to_yaml(self.cfg, resolve=True), encoding="utf-8"
        )
        self.apply_seed()
        return self

    def is_complete(self) -> bool:
        """True si le manifeste existe : le run est rejouable et agregeable."""
        return self.paths.manifest(self.run_id).exists()

    def write_manifest(self, **fields: Any) -> Path:
        """Ecrit `manifest.json` EN DERNIER (contrat 5, §3.5).

        Effet de bord : ecrit runs/<run_id>/manifest.json.
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
                # Le materiel fait partie des conditions de comparaison (ADR-0019).
                "device": device_info(self.cfg.train.get("device", "auto")),
            },
            "config": resolved,
            **self.extra,
            **fields,
        }
        return write_json(self.paths.manifest(self.run_id), manifest)


def _package_versions() -> dict[str, str]:
    """Versions des dependances qui influencent les resultats."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "torch", "ultralytics", "optuna", "scikit-learn", "peft"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out
