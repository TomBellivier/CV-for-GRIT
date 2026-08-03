"""Tests des approches D (lora), E (group_bn) et F (yolo_pooled_reduced).

Ces trois approches modifient le `nn.Module` construit par Ultralytics. Le patch
lui-meme exige torch et ne peut pas etre teste sans GPU ni dependance lourde. En
revanche, **toute la logique de decision** — quels modules recoivent un adaptateur,
quels parametres sont geles, a quel groupe appartient une image, quels keypoints sont
retires — est ecrite sous forme de fonctions pures, et c'est elle qui est testee ici.

C'est deliberement la partie qui casse en silence : un motif de selection errone donne
un entrainement qui tourne normalement et n'apprend rien.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from insectpose import pipeline
from insectpose.approaches.yolo_reduced import dropped_indices, mask_keypoints
from insectpose.cli import load_config
from insectpose.data.keypoints import load_schema
from insectpose.models.group_norm import (
    CONTEXT,
    active_group,
    dataset_indices_from_paths,
    default_datasets,
)
from insectpose.registry import APPROACHES
from insectpose.training.patching import (
    freeze_patterns_for,
    head_index,
    match_module_names,
)
from insectpose.utils.io import read_json, read_parquet

SCHEMA = "insect42_v1"

# Structure typique d'un YOLO-pose : Sequential de blocs, la tete en dernier.
MODULE_NAMES = [
    "model", "model.0", "model.0.conv", "model.0.bn",
    "model.9", "model.9.conv", "model.9.bn",
    "model.20", "model.20.cv1.conv", "model.20.cv2.conv",
    "model.21", "model.21.conv", "model.22", "model.22.conv",
    "model.23", "model.23.cv4.0.0.conv",          # tete
]


# ===========================================================================
# Selection des modules (approche D)
# ===========================================================================
def test_head_index_is_the_last_block() -> None:
    assert head_index(MODULE_NAMES) == 23


def test_head_index_refuses_unexpected_structure() -> None:
    with pytest.raises(ValueError, match="structure inattendue"):
        head_index(["backbone.conv", "neck.conv"])


def test_lora_targets_are_the_last_neck_blocks() -> None:
    """Les index sont calcules depuis la structure : changer de taille de reseau ne casse rien."""
    last = head_index(MODULE_NAMES)
    blocks = "|".join(str(i) for i in range(last - 3, last))
    targets = match_module_names(MODULE_NAMES, [rf"^model\.({blocks})\..*\bconv$"])
    assert set(targets) == {"model.20.cv1.conv", "model.20.cv2.conv",
                            "model.21.conv", "model.22.conv"}
    assert not any(t.startswith("model.23") for t in targets)   # la tete est exclue
    assert not any(t.startswith("model.0") for t in targets)    # le backbone aussi


def test_freeze_keeps_adapters_and_head_trainable() -> None:
    """Le gel doit epargner les adaptateurs, sinon l'entrainement n'apprend rien."""
    parameters = [
        "model.0.conv.weight",
        "model.20.cv1.conv.base_layer.weight",
        "model.20.cv1.conv.lora_A.default.weight",
        "model.20.cv1.conv.lora_B.default.weight",
        "model.23.cv4.0.0.conv.weight",
    ]
    frozen = freeze_patterns_for(parameters, [r"lora_[AB]", r"^model\.23\."])
    assert frozen == ["model.0.conv.weight", "model.20.cv1.conv.base_layer.weight"]


def test_freeze_without_head_keeps_only_adapters() -> None:
    parameters = ["model.0.conv.weight", "model.20.cv1.conv.lora_A.default.weight",
                  "model.23.cv4.0.0.conv.weight"]
    assert freeze_patterns_for(parameters, [r"lora_[AB]"]) == [
        "model.0.conv.weight", "model.23.cv4.0.0.conv.weight"]


def test_lora_declares_its_dependency() -> None:
    available, reason = APPROACHES.get("lora").availability()
    assert available or "peft" in reason or "torch" in reason


def test_lora_config_records_what_matters(project) -> None:
    """Ce qui reste degele a cote des adaptateurs est LA variable cachee (ADR-0025)."""
    cfg = load_config([f"paths.root={project.root}", "approach=lora"])
    assert int(cfg.approach.lora.r) > 0
    assert bool(cfg.approach.lora.train_head) is True
    assert "lora.r" in dict(cfg.approach.search_space)


# ===========================================================================
# Routage par groupe (approche E)
# ===========================================================================
def test_dataset_index_is_read_from_exported_filenames() -> None:
    """L'export YOLO aplatit `<dataset>/<stem>` : le prefixe porte le groupe."""
    paths = ["/x/y/coleoptera__img001.jpg", "/x/y/diptera__img002.jpg",
             "hymenoptera__img003.png"]
    datasets = ["coleoptera", "diptera", "hymenoptera", "lepidoptera"]
    assert list(dataset_indices_from_paths(paths, datasets)) == [0, 1, 2]


def test_unknown_dataset_is_an_explicit_error() -> None:
    """ADR-0014 : le groupe est toujours declare ; l'inconnu ne se devine pas."""
    with pytest.raises(RuntimeError, match="Dataset indeterminable"):
        dataset_indices_from_paths(["/x/orthoptera__img001.jpg"], ["coleoptera"])


def test_missing_group_context_is_an_explicit_error() -> None:
    CONTEXT.clear()
    with pytest.raises(RuntimeError, match="Aucun groupe"):
        CONTEXT.require(4)


def test_single_group_is_broadcast_to_the_batch() -> None:
    with active_group(2):
        assert list(CONTEXT.require(3)) == [2, 2, 2]
    assert CONTEXT.indices is None


def test_group_context_size_mismatch_is_refused() -> None:
    with active_group(np.array([0, 1])), pytest.raises(RuntimeError, match="lot de 5"):
        CONTEXT.require(5)


def test_group_order_follows_the_frozen_dataset_order(cfg) -> None:
    """L'ordre des groupes doit etre stable : il indexe des poids sauvegardes."""
    OmegaConf.update(cfg, "data.datasets", ["diptera", "coleoptera"])
    assert default_datasets(cfg) == ["coleoptera", "diptera"]


# ===========================================================================
# Retrait de keypoints (approche F)
# ===========================================================================
def test_dropped_keypoints_cover_legs_and_hindwings(project) -> None:
    schema = load_schema(SCHEMA, project.configs)
    indices = dropped_indices(schema, ["leg", "hindwing"])
    assert len(indices) == 16
    assert schema.index("left-leg-0") in indices
    assert schema.index("right-hindwing-tip") in indices
    assert schema.index("head-top") not in indices
    assert schema.index("left-forewing-tip") not in indices


def test_masking_sets_visibility_to_zero_without_moving_points() -> None:
    """Un point masque n'est pas appris ; ses coordonnees restent intactes."""
    annotations = pd.DataFrame([{"kpts_vis": [2, 2, 1, 2], "kpts_xy": [1.0, 2.0] * 4}])
    masked = mask_keypoints(annotations, [1, 3])
    assert masked["kpts_vis"].iloc[0] == [2, 0, 1, 0]
    assert masked["kpts_xy"].iloc[0] == annotations["kpts_xy"].iloc[0]
    assert annotations["kpts_vis"].iloc[0] == [2, 2, 1, 2]   # original non modifie


def test_reduced_approach_masks_train_and_val_but_not_test(
    fake_ultralytics, config_factory, project  # noqa: ARG001
) -> None:
    """Le test reste intact : c'est la verite terrain commune a toutes les approches."""
    cfg = config_factory(["approach=yolo_pooled_reduced", "train.device=cpu"])
    pipeline.cmd_split(cfg)
    ctx, data, approach = pipeline._prepare_run(cfg)
    ctx.setup()
    prepared = approach._prepare_data(data, ctx)

    schema = load_schema(SCHEMA, project.configs)
    dropped = dropped_indices(schema, ["leg", "hindwing"])
    for role in ("train", "val"):
        vis = np.stack(prepared.role(role).annotations["kpts_vis"].map(np.asarray).to_numpy())
        assert (vis[:, dropped] == 0).all()
    vis_test = np.stack(prepared.test.annotations["kpts_vis"].map(np.asarray).to_numpy())
    assert (vis_test[:, dropped] > 0).any(), "le test ne doit pas etre masque"


def test_reduced_approach_records_what_was_dropped(
    fake_ultralytics, config_factory, project  # noqa: ARG001
) -> None:
    cfg = config_factory(["approach=yolo_pooled_reduced", "train.device=cpu"])
    pipeline.cmd_split(cfg)
    ctx = pipeline.cmd_train(cfg)
    manifest = read_json(project.manifest(ctx.run_id))
    assert manifest["n_supervised_keypoints"] == 26
    assert len(manifest["dropped_keypoints"]) == 16

    # Les predictions restent au schema complet : le contrat 3 l'exige (§3.4)
    predictions = read_parquet(project.predictions(ctx.run_id, "test", ctx.fold),
                               artifact="predictions", validate=True)
    assert len(predictions["kpts_xy"].iloc[0]) == 84


def test_unmatched_drop_pattern_is_refused(fake_ultralytics, config_factory) -> None:  # noqa: ARG001
    """Un motif qui ne correspond a rien rendrait F identique a A, en silence."""
    cfg = config_factory(["approach=yolo_pooled_reduced", "train.device=cpu"])
    OmegaConf.update(cfg, "approach.drop_keypoints", ["inexistant"])
    pipeline.cmd_split(cfg)
    with pytest.raises(ValueError, match="Aucun keypoint ne correspond"):
        pipeline.cmd_train(cfg)


# ===========================================================================
# Comparaison equitable de F (ADR-0027)
# ===========================================================================
def test_excluded_keypoints_produce_a_retained_mean(reported) -> None:
    """La comparaison valide de F porte sur les points CONSERVES."""
    from insectpose.reporting.compare import CompareFilter, write_comparison

    _, project, _ = reported
    produced = write_comparison(
        project, CompareFilter(exclude_keypoints=("leg", "hindwing")))
    assert any(p.name.startswith("heatmap_keypoints") for p in produced)


@pytest.fixture()
def reported(cfg, project):
    from insectpose.evaluation.aggregate import write_master

    pipeline.cmd_split(cfg)
    pipeline.cmd_train(cfg)
    return cfg, project, read_parquet(write_master(project))


# ===========================================================================
# Remplacement des modules (approche E) - testable sans torch
# ===========================================================================
class _FakeModule:
    """Arbre de modules minimal, imitant l'interface `named_children` de torch."""

    def __init__(self, **children: object) -> None:
        for name, child in children.items():
            setattr(self, name, child)
        self._names = list(children)

    def named_children(self):
        return [(name, getattr(self, name)) for name in self._names]


class _FakeBN:
    pass


class _FakeGroupBN:
    """Remplacant contenant lui-meme des BN : c'est ce qui provoquait la recursion."""

    def __init__(self, source: object, n: int = 4) -> None:
        self.source = source
        self.branches = _FakeModule(**{f"b{i}": _FakeBN() for i in range(n)})
        self._names = ["branches"]

    def named_children(self):
        return [(name, getattr(self, name)) for name in self._names]


def _replace(root):
    from insectpose.models.group_norm import replace_modules

    return replace_modules(
        root,
        is_target=lambda m: isinstance(m, _FakeBN),
        make_replacement=lambda m: _FakeGroupBN(m),
        is_replacement=lambda m: isinstance(m, _FakeGroupBN),
    )


def test_replacement_does_not_recurse_into_its_own_output() -> None:
    """Regression : le remplacant contient des BN ; les remplacer donnerait une recursion."""
    root = _FakeModule(a=_FakeBN(), b=_FakeModule(c=_FakeBN(), d=_FakeModule(e=_FakeBN())))
    assert _replace(root) == 3
    assert isinstance(root.a, _FakeGroupBN)
    assert isinstance(root.b.c, _FakeGroupBN)
    assert isinstance(root.b.d.e, _FakeGroupBN)
    # Les BN internes du remplacant restent intactes
    assert isinstance(root.a.branches.b0, _FakeBN)


def test_replacement_is_idempotent() -> None:
    """Un second passage ne doit rien remplacer : sinon l'imbrication recommence."""
    root = _FakeModule(a=_FakeBN(), b=_FakeModule(c=_FakeBN()))
    assert _replace(root) == 2
    assert _replace(root) == 0


def test_replacement_counts_zero_on_a_model_without_batchnorm() -> None:
    """Zero remplacement = approche sans effet : l'appelant doit lever une erreur."""
    assert _replace(_FakeModule(a=_FakeModule(), b=_FakeModule())) == 0


# ===========================================================================
# Trainer patche : validateur, evaluation finale (testable sans torch)
# ===========================================================================
class _FakeValidator:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def preprocess(self, batch: object) -> object:
        self.seen.append(batch)
        return batch


class _FakeTrainerBase:
    """Base minimale imitant l'interface du trainer Ultralytics utilisee par le patch."""

    def __init__(self) -> None:
        self.model = _FakeModule()
        self.final_eval_called = False
        self.validator = _FakeValidator()

    def get_model(self, cfg=None, weights=None, verbose=True):  # noqa: ANN001, ANN201, ARG002
        return self.model

    def get_validator(self) -> _FakeValidator:
        return self.validator

    def preprocess_batch(self, batch: object) -> object:
        return batch

    def _build_train_pipeline(self) -> str:
        return "built"

    def final_eval(self) -> str:
        self.final_eval_called = True
        return "evaluated"


def test_validator_batches_also_update_the_context() -> None:
    """Le validateur a son propre preprocess : sans relais, le contexte reste perime."""
    from insectpose.training.patching import make_patched_trainer

    seen: list[object] = []
    cls = make_patched_trainer(_FakeTrainerBase, on_batch=lambda _t, b: seen.append(b))
    trainer = cls()

    trainer.preprocess_batch({"im_file": ["a"]})
    trainer.get_validator().preprocess({"im_file": ["b", "c"]})
    assert seen == [{"im_file": ["a"]}, {"im_file": ["b", "c"]}]


def test_final_eval_can_be_skipped() -> None:
    """Sur un modele patche, l'evaluation finale rechargerait et fusionnerait le modele."""
    from insectpose.training.patching import make_patched_trainer

    skipping = make_patched_trainer(_FakeTrainerBase, skip_final_eval=True)()
    assert skipping.final_eval() is None
    assert not skipping.final_eval_called

    normal = make_patched_trainer(_FakeTrainerBase)()
    assert normal.final_eval() == "evaluated"


def test_patch_is_applied_at_model_construction() -> None:
    from insectpose.training.patching import make_patched_trainer

    patched: list[object] = []
    cls = make_patched_trainer(_FakeTrainerBase, patch=patched.append)
    trainer = cls()
    assert trainer.get_model() is trainer.model
    assert patched == [trainer.model]


def test_lora_merge_replaces_wrappers_by_plain_convolutions() -> None:
    """Apres fusion, le checkpoint ne depend plus de peft et redevient fusionnable."""
    from insectpose.models.group_norm import replace_modules

    class _FakeLora:
        def __init__(self, base: object) -> None:
            self.base = base
            self.merged = False

        def named_children(self):
            return [("base", self.base)]

        def merge(self) -> None:
            self.merged = True

        def get_base_layer(self) -> object:
            return self.base

    conv = object()
    root = _FakeModule(a=_FakeLora(conv))
    wrapper = root.a
    merged = replace_modules(
        root,
        is_target=lambda m: isinstance(m, _FakeLora),
        make_replacement=lambda m: (m.merge(), m.get_base_layer())[1],
        is_replacement=lambda _m: False,
    )
    assert merged == 1
    assert root.a is conv
    assert wrapper.merged


# ===========================================================================
# Serialisation des classes construites dynamiquement (approche E)
# ===========================================================================
def _make_local_class():
    """Classe definie dans une fonction : introuvable par pickle en l'etat."""

    class Dynamic:
        def __init__(self, value: int = 1) -> None:
            self.value = value

    return Dynamic


def test_dynamic_class_is_not_picklable_without_registration() -> None:
    """Regression : Ultralytics serialise le modele a chaque sauvegarde de checkpoint."""
    import pickle

    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(_make_local_class()())


def test_registration_makes_a_dynamic_class_picklable() -> None:
    import pickle

    from insectpose.models.group_norm import register_picklable

    cls = register_picklable(_make_local_class(), globals(), qualname="RegisteredDynamic")
    restored = pickle.loads(pickle.dumps(cls(7)))
    assert restored.value == 7
    assert cls.__module__ == __name__
    assert globals()["RegisteredDynamic"] is cls


def test_group_batchnorm_is_exposed_at_module_level() -> None:
    """Pickle resout `module.GroupBatchNorm2d` : l'attribut doit exister ou echouer clairement."""
    import insectpose.models.group_norm as module

    try:
        cls = module.GroupBatchNorm2d
    except ImportError as exc:          # torch absent : message actionnable attendu
        assert "torch" in str(exc)
        return
    assert cls.__qualname__ == "GroupBatchNorm2d"
    assert cls.__module__ == "insectpose.models.group_norm"


def test_unknown_module_attribute_still_raises() -> None:
    import insectpose.models.group_norm as module

    attribute = "inexistant"   # nom variable : sinon ruff exige l'acces direct
    with pytest.raises(AttributeError, match="inexistant"):
        getattr(module, attribute)


# ===========================================================================
# Convolutions groupees (approche D)
# ===========================================================================
CONVOLUTIONS = [
    ("model.0.conv", 1),
    ("model.20.cv1.conv", 1),
    ("model.20.cv2.conv", 128),      # depthwise : peft exige rang % groups == 0
    ("model.21.conv", 1),
    ("model.22.conv", 256),          # depthwise
    ("model.23.cv4.0.0.conv", 1),    # tete
]


def test_grouped_convolutions_are_excluded_from_lora_targets() -> None:
    """Regression : peft refuse une depthwise si le rang n'est pas divisible par groups."""
    from insectpose.training.patching import match_conv_targets

    kept, skipped = match_conv_targets(CONVOLUTIONS, [r"^model\.(20|21|22)\..*\bconv$"])
    assert kept == ["model.20.cv1.conv", "model.21.conv"]
    assert skipped == ["model.20.cv2.conv", "model.22.conv"]
    assert "model.0.conv" not in kept + skipped        # hors motif
    assert "model.23.cv4.0.0.conv" not in kept         # la tete n'est pas adaptee


def test_all_grouped_targets_leaves_nothing_to_adapt() -> None:
    """Le cas doit etre detectable : sinon l'entrainement n'adapterait rien."""
    from insectpose.training.patching import match_conv_targets

    kept, skipped = match_conv_targets([("model.22.conv", 256)], [r"^model\.22\."])
    assert kept == []
    assert len(skipped) == 1