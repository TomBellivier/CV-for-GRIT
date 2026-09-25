"""BatchNorm par groupe d'insecte (ADR-0026).

Chaque BatchNorm du reseau est remplacee par N copies, une par dataset : statistiques
courantes ET parametres affines. Les poids convolutifs restent partages, seule la
normalisation est conditionnee — c'est l'hypothese testee par l'approche E.

Les lots sont **mixtes** (les 4 datasets melanges) : le forward se scinde par groupe,
puis recompose dans l'ordre d'origine. Cela evite la correlation entre le gradient et
le dataset qu'introduiraient des lots homogenes.

Le groupe courant est porte par un contexte de module, renseigne :
- a l'entrainement, depuis les noms de fichiers du lot (l'export YOLO les prefixe par
  le dataset, cf. `yolo_export.flat_name`) ;
- a l'inference, explicitement, puisque l'utilisateur declare toujours l'ordre traite
  (ADR-0014). Un groupe inconnu est une **erreur explicite**, jamais un repli devine.
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
    """Groupe(s) actif(s) pour le prochain forward.

    Volontairement global au processus : les modules de normalisation sont appeles au
    plus profond du reseau, la ou aucune information de dataset ne circule.
    """

    def __init__(self) -> None:
        self.indices: Any = None

    def set(self, indices: Any) -> None:
        self.indices = indices

    def clear(self) -> None:
        self.indices = None

    def require(self, batch_size: int) -> Any:
        """Indices de groupe du lot courant, ou echec explicite."""
        if self.indices is None:
            raise RuntimeError(
                "Aucun groupe d'insecte declare avant le forward. Les modeles a "
                "normalisation par groupe exigent de connaitre l'ordre traite "
                "(ADR-0014) : utiliser `active_group(...)` ou renseigner le contexte."
            )
        indices = np.atleast_1d(np.asarray(self.indices, dtype=int))
        if indices.size == 1:
            return np.repeat(indices, batch_size)
        if indices.size != batch_size:
            raise RuntimeError(
                f"{indices.size} indice(s) de groupe pour un lot de {batch_size} : "
                "le contexte n'a pas ete mis a jour pour ce lot."
            )
        return indices


CONTEXT = GroupContext()


@contextmanager
def active_group(indices: Any) -> Any:
    """Fixe le groupe actif le temps d'un bloc, puis le libere."""
    previous = CONTEXT.indices
    CONTEXT.set(indices)
    try:
        yield
    finally:
        CONTEXT.set(previous)


def dataset_indices_from_paths(paths: list[str], datasets: list[str]) -> np.ndarray:
    """Indices de dataset deduits des noms de fichiers exportes.

    L'export YOLO aplatit `<dataset>/<stem>` en `<dataset>__<stem>` : le prefixe est
    donc porte par le nom de fichier. Fonction pure, testable sans torch.
    """
    lookup = {name: i for i, name in enumerate(datasets)}
    indices = []
    for path in paths:
        stem = str(path).replace("\\", "/").split("/")[-1]
        match = re.match(r"([A-Za-z0-9]+)__", stem)
        name = match.group(1) if match else None
        if name not in lookup:
            raise RuntimeError(
                f"Dataset indeterminable pour '{stem}'. Les modeles a normalisation par "
                f"groupe exigent un dataset connu parmi {datasets} (ADR-0014)."
            )
        indices.append(lookup[name])
    return np.asarray(indices, dtype=int)


def register_picklable(cls: Any, namespace: dict[str, Any], qualname: str | None = None) -> Any:
    """Rend picklable une classe creee dynamiquement.

    Pickle ne serialise pas le code d'une classe : il enregistre son chemin
    (`module.QualName`) et le resout a la lecture. Une classe definie DANS une fonction
    est donc introuvable — et Ultralytics serialise le modele a chaque sauvegarde de
    checkpoint. On corrige son identite et on la publie dans le module.

    Fonction pure : testable sans torch.
    """
    name = qualname or cls.__name__
    cls.__module__ = namespace["__name__"]
    cls.__qualname__ = name
    namespace[name] = cls
    return cls


_GROUP_BN_CLASS: Any = None


def build_group_batchnorm() -> Any:
    """Fabrique (une seule fois) la classe GroupBatchNorm2d, import torch differe."""
    global _GROUP_BN_CLASS
    if _GROUP_BN_CLASS is not None:
        return _GROUP_BN_CLASS
    torch = require("torch", "dev")
    nn = torch.nn

    class GroupBatchNorm2d(nn.Module):  # type: ignore[misc, valid-type]
        """N BatchNorm2d paralleles, une par dataset, routees par le contexte."""

        def __init__(self, source: Any, n_groups: int) -> None:
            super().__init__()
            self.n_groups = n_groups
            self.branches = nn.ModuleList([
                nn.BatchNorm2d(source.num_features, eps=source.eps,
                               momentum=source.momentum, affine=source.affine,
                               track_running_stats=source.track_running_stats)
                for _ in range(n_groups)
            ])
            # Chaque branche part des statistiques et des affines du modele pre-entraine :
            # la specialisation commence donc d'un point commun, pas d'une initialisation
            # aleatoire qui detruirait les poids COCO.
            for branch in self.branches:
                branch.load_state_dict(source.state_dict())
            self.num_features = source.num_features

        def forward(self, x: Any) -> Any:
            indices = CONTEXT.require(x.shape[0])
            unique = np.unique(indices)
            if unique.size == 1:
                return self.branches[int(unique[0])](x)
            # Lot mixte : on scinde par groupe puis on recompose dans l'ordre d'origine.
            output = torch.empty_like(x)
            for group in unique:
                mask = torch.as_tensor(indices == group, device=x.device)
                output[mask] = self.branches[int(group)](x[mask])
            return output

    _GROUP_BN_CLASS = register_picklable(GroupBatchNorm2d, globals())
    return _GROUP_BN_CLASS


def __getattr__(name: str) -> Any:
    """Construit `GroupBatchNorm2d` a la demande (PEP 562).

    Permet a pickle de resoudre `insectpose.models.group_norm.GroupBatchNorm2d` au
    chargement d'un checkpoint, meme si la classe n'a pas encore ete construite dans ce
    processus, tout en gardant l'import de torch differe.
    """
    if name == "GroupBatchNorm2d":
        return build_group_batchnorm()
    raise AttributeError(f"module {__name__!r} n'a pas d'attribut {name!r}")


def replace_modules(root: Any, is_target: Any, make_replacement: Any,
                    is_replacement: Any) -> int:
    """Remplace en profondeur les modules cibles, sans descendre dans les remplacants.

    Deux precautions, toutes deux indispensables :
    - le parcours materialise `named_children()` en liste avant de modifier l'arbre ;
      iterer un generateur que l'on mute donne un comportement indefini ;
    - il ne descend PAS dans un module deja remplace. Un `GroupBatchNorm2d` contient
      lui-meme N `BatchNorm2d` : sans ce garde-fou, elles seraient remplacees a leur
      tour, indefiniment, jusqu'a une RecursionError.

    Fonction pure vis-a-vis de torch : les predicats sont injectes, donc testable.
    Retourne le nombre de remplacements.
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
    """Remplace toutes les BatchNorm2d du modele par des versions par groupe.

    Retourne le nombre de couches remplacees. Zero signalerait un modele sans BN, donc
    une approche sans effet : l'appelant doit le traiter comme une erreur.
    """
    torch = require("torch", "dev")
    group_cls = build_group_batchnorm()

    replaced = replace_modules(
        model,
        is_target=lambda m: isinstance(m, torch.nn.BatchNorm2d),
        make_replacement=lambda m: group_cls(m, n_groups),
        is_replacement=lambda m: isinstance(m, group_cls),
    )
    log.info("%d BatchNorm2d remplacees par des versions a %d groupes.", replaced, n_groups)
    return replaced


def default_datasets(cfg: Any) -> list[str]:
    """Datasets du perimetre courant, dans l'ordre fige de `contracts.DATASETS`."""
    wanted = {str(d) for d in cfg.data.datasets}
    return [name for name in DATASETS if name in wanted]