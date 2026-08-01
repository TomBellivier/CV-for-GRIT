"""Tests de la resolution du peripherique de calcul (ADR-0019).

Le materiel fait partie des conditions d'une comparaison : il doit etre resolu
explicitement, journalise, et enregistre dans le manifeste.
"""

from __future__ import annotations

import pytest

from insectpose.utils.device import (
    amp_enabled,
    device_indices,
    device_info,
    peak_vram_mb,
    resolve_device,
)


def test_auto_resolves_to_gpu_or_cpu() -> None:
    resolved = resolve_device("auto")
    assert resolved == "cpu" or resolved.split(",")[0].isdigit()


def test_explicit_device_is_respected() -> None:
    """Demander 'cpu' sur une machine a GPU est un choix legitime, pas une erreur."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("0,1") == "0,1"
    assert resolve_device(0) == "0"


def test_device_indices_parses_multi_gpu() -> None:
    assert device_indices("0,1") == [0, 1]
    assert device_indices("cpu") == []
    assert device_indices("mps") == []


def test_device_info_is_serialisable(cfg) -> None:
    import json

    info = device_info(cfg.train.device)
    assert {"requested", "resolved", "cuda_available"} <= set(info)
    json.dumps(info)   # doit pouvoir entrer tel quel dans le manifeste


def test_amp_is_disabled_in_debug_mode() -> None:
    """En mode debug, la reproductibilite prime sur la vitesse (§6.4)."""
    assert not amp_enabled(True, "debug", "0")
    assert not amp_enabled(True, "full", "cpu")   # sans effet hors CUDA
    assert not amp_enabled(False, "full", "0")
    assert amp_enabled(True, "full", "0")


def test_peak_vram_is_none_without_cuda() -> None:
    value = peak_vram_mb()
    assert value is None or value >= 0


def test_manifest_records_hardware(cfg, project) -> None:
    from insectpose import pipeline
    from insectpose.utils.io import read_json

    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    environment = read_json(project.manifest(ctx.run_id))["environment"]
    assert "device" in environment
    assert "cuda_available" in environment["device"]


@pytest.mark.parametrize("spec", ["auto", "cpu", "0"])
def test_resolution_is_idempotent(spec: str) -> None:
    assert resolve_device(resolve_device(spec)) == resolve_device(spec)
