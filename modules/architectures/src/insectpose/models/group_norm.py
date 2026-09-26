"""BatchNorm per insect group (ADR-0026).

Each BatchNorm of the network is replaced by N copies, one per dataset: running
statistics AND affine parameters. The convolution weights stay shared, only the
normalisation is conditioned — that is the hypothesis tested by approach E.

Batches are **mixed** (the 4 datasets together): the forward pass splits by group, then
recomposes in the original order. This avoids the correlation between the gradient and
the dataset that homogeneous batches would introduce.

The current group is carried by a module context, filled:
- at training time, from the file names of the batch (the YOLO export prefixes them with
  the dataset, see `yolo_export.flat_name`);
- at inference, explicitly, since the user always declares the order processed
  (ADR-0014). An unknown group is an **explicit error**, never a guessed fallback.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

import numpy as np

from insectpose.contracts import DATASETS
from insectpose.utils.logging import get_logger
from insectpose.utils.optional import require

log = get_logger("group_norm")


class GroupContext:
    """Active group(s) for the next forward pass.

    Deliberately global to the process: the normalisation modules are called deep inside
    the network, where no dataset information flows.
    """

    def __init__(self) -> None:
        self.indices: Any = None

    def set(self, indices: Any) -> None:
        self.indices = indices

    def clear(self) -> None:
        self.indices = None

    def require(self, batch_size: int) -> Any:
        """Group indices of the current batch, or an explicit failure."""
        if self.indices is None:
            raise RuntimeError(
                "No insect group declared before the forward pass. The per-group "
                "normalisation models require the order processed to be known "
                "(ADR-0014): use `active_group(...)` or fill the context."
            )
        indices = np.atleast_1d(np.asarray(self.indices, dtype=int))
        if indices.size == 1:
            return np.repeat(indices, batch_size)
        if indices.size != batch_size:
            raise RuntimeError(
                f"{indices.size} group index(es) for a batch of {batch_size}: "
                "the context was not updated for this batch."
            )
        return indices


CONTEXT = GroupContext()


@contextmanager
def active_group(indices: Any) -> Any:
    """Set the active group for the duration of a block, then release it."""
    previous = CONTEXT.indices
    CONTEXT.set(indices)
    try:
        yield
    finally:
        CONTEXT.set(previous)


def dataset_indices_from_paths(paths: list[str], datasets: list[str]) -> np.ndarray:
    """Dataset indices derived from the exported file names.

    The YOLO export flattens `<dataset>/<stem>` into `<dataset>__<stem>`: the prefix is
    therefore carried by the file name. Pure function, testable without torch.
    """
    lookup = {name: i for i, name in enumerate(datasets)}
    indices = []
    for path in paths:
        stem = str(path).replace("\\", "/").split("/")[-1]
        match = re.match(r"([A-Za-z0-9]+)__", stem)
        name = match.group(1) if match else None
        if name not in lookup:
            raise RuntimeError(
                f"Cannot determine the dataset of '{stem}'. The per-group normalisation "
                f"models require a known dataset among {datasets} (ADR-0014)."
            )
        indices.append(lookup[name])
    return np.asarray(indices, dtype=int)


def register_picklable(cls: Any, namespace: dict[str, Any], qualname: str | None = None) -> Any:
    """Make a dynamically created class picklable.

    Pickle does not serialise the code of a class: it records its path
    (`module.QualName`) and resolves it when reading. A class defined INSIDE a function
    therefore cannot be found — and Ultralytics serialises the model at every checkpoint
    save. Its identity is fixed and it is published in the module.

    Pure function: testable without torch.
    """
    name = qualname or cls.__name__
    cls.__module__ = namespace["__name__"]
    cls.__qualname__ = name
    namespace[name] = cls
    return cls


_GROUP_BN_CLASS: Any = None


def build_group_batchnorm() -> Any:
    """Build (only once) the GroupBatchNorm2d class, deferred torch import."""
    global _GROUP_BN_CLASS
    if _GROUP_BN_CLASS is not None:
        return _GROUP_BN_CLASS
    torch = require("torch", "dev")
    nn = torch.nn

    class GroupBatchNorm2d(nn.Module):  # type: ignore[misc, valid-type]
        """N parallel BatchNorm2d, one per dataset, routed by the context."""

        def __init__(self, source: Any, n_groups: int) -> None:
            super().__init__()
            self.n_groups = n_groups
            self.branches = nn.ModuleList([
                nn.BatchNorm2d(source.num_features, eps=source.eps,
                               momentum=source.momentum, affine=source.affine,
                               track_running_stats=source.track_running_stats)
                for _ in range(n_groups)
            ])
            # Each branch starts from the statistics and affines of the pre-trained model:
            # the specialisation therefore starts from a common point, not from a random
            # initialisation that would destroy the COCO weights.
            for branch in self.branches:
                branch.load_state_dict(source.state_dict())
            self.num_features = source.num_features

        def forward(self, x: Any) -> Any:
            indices = CONTEXT.require(x.shape[0])
            unique = np.unique(indices)
            if unique.size == 1:
                return self.branches[int(unique[0])](x)
            # Mixed batch: split by group then recompose in the original order.
            output = torch.empty_like(x)
            for group in unique:
                mask = torch.as_tensor(indices == group, device=x.device)
                output[mask] = self.branches[int(group)](x[mask])
            return output

    _GROUP_BN_CLASS = register_picklable(GroupBatchNorm2d, globals())
    return _GROUP_BN_CLASS


def __getattr__(name: str) -> Any:
    """Build `GroupBatchNorm2d` on demand (PEP 562).

    Lets pickle resolve `insectpose.models.group_norm.GroupBatchNorm2d` when loading a
    checkpoint, even if the class has not been built yet in this process, while keeping
    the torch import deferred.
    """
    if name == "GroupBatchNorm2d":
        return build_group_batchnorm()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def replace_modules(root: Any, is_target: Any, make_replacement: Any,
                    is_replacement: Any) -> int:
    """Replace the target modules in depth, without going down into the replacements.

    Two precautions, both essential:
    - the traversal materialises `named_children()` into a list before modifying the
      tree; iterating a generator being mutated gives an undefined behaviour;
    - it does NOT go down into an already replaced module. A `GroupBatchNorm2d` itself
      contains N `BatchNorm2d`: without this guard, they would be replaced in turn,
      indefinitely, until a RecursionError.

    Pure function with respect to torch: the predicates are injected, hence testable.
    Returns the number of replacements.
    """
    replaced = 0
    for name, child in list(root.named_children()):
        if is_target(child):
            setattr(root, name, make_replacement(child))
            replaced += 1
        elif not is_replacement(child):
            replaced += replace_modules(child, is_target, make_replacement, is_replacement)
    return replaced


def replace_batchnorm(model: Any, n_groups: int) -> int:
    """Replace every BatchNorm2d of the model by per-group versions.

    Returns the number of layers replaced. Zero would signal a model without BN, hence an
    approach without effect: the caller must treat it as an error.
    """
    torch = require("torch", "dev")
    group_cls = build_group_batchnorm()

    replaced = replace_modules(
        model,
        is_target=lambda m: isinstance(m, torch.nn.BatchNorm2d),
        make_replacement=lambda m: group_cls(m, n_groups),
        is_replacement=lambda m: isinstance(m, group_cls),
    )
    log.info("%d BatchNorm2d replaced by %d-group versions.", replaced, n_groups)
    return replaced


def default_datasets(cfg: Any) -> list[str]:
    """Datasets of the current scope, in the frozen order of `contracts.DATASETS`."""
    wanted = {str(d) for d in cfg.data.datasets}
    return [name for name in DATASETS if name in wanted]