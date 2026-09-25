"""Patch du modele Ultralytics avant entrainement (ADR-0025, ADR-0026, ADR-0028).

Ultralytics ne prevoit ni adaptateurs LoRA ni normalisation conditionnelle. Les deux
approches doivent donc modifier le `nn.Module` construit par le trainer. Ce module
isole tout ce qui depend des internes d'Ultralytics, pour qu'une mise a jour de la
bibliotheque ne casse qu'un seul endroit.

Trois internes sont utilises, verifies sur les sources de la version installee :

1. **Les callbacks ne conviennent pas.** `on_pretrain_routine_start` se declenche AVANT
   la construction du modele ; `on_pretrain_routine_end` APRES la creation de
   l'optimiseur et de l'EMA. Un patch pose a ces moments serait soit perdu, soit absent
   de l'optimiseur. On passe donc un trainer personnalise (`train(trainer=...)`), et le
   patch est applique dans `get_model`, au moment meme de la construction.

2. **Ultralytics degele ce que l'on gele.** Sa boucle de `freeze` remet
   `requires_grad=True` sur tout parametre gele dont le nom ne correspond pas a
   `args.freeze`, en emettant "setting 'requires_grad=True' for frozen layer '...'".
   Un simple `requires_grad=False` est donc silencieusement annule.

   **L'ordre reel compte** : cette boucle vit dans `_setup_train`, qui appelle ENSUITE
   `_build_train_pipeline` pour construire l'optimiseur. Reappliquer le gel dans
   `_build_train_pipeline` — comme le faisait une version anterieure de ce module —
   arrivait donc AVANT le degel, et etait annule : l'entrainement mettait a jour tout
   le reseau en se presentant comme du LoRA. Le gel est desormais applique a la SORTIE
   de `_setup_train`, apres le degel.

3. **Le pipeline peut etre reconstruit en cours d'entrainement** (changement de taille
   de lot, reprise), ce qui relancerait la boucle de degel. Le gel est donc reapplique
   a chaque epoque via `_model_train`.

Le compte de parametres entrainables est journalise et enregistre au manifeste : c'est
le seul signal fiable qu'un patch a pris. Un ratio proche de 1 sur une approche LoRA
signale immediatement que le gel n'a pas fonctionne.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from insectpose.utils.logging import get_logger

log = get_logger("patching")

PatchFn = Callable[[Any], None]


# --- selection de modules : pur, testable sans torch -------------------------
def match_module_names(names: Iterable[str], patterns: Iterable[str]) -> list[str]:
    """Noms de modules correspondant a au moins une expression reguliere.

    Fonction pure : c'est elle qui decide OU vont les adaptateurs, et c'est donc elle
    qu'il faut tester. Le reste du patch n'est que de la plomberie torch.
    """
    compiled = [re.compile(p) for p in patterns]
    return [name for name in names if any(c.search(name) for c in compiled)]


def match_conv_targets(convolutions: Iterable[tuple[str, int]],
                       patterns: Iterable[str]) -> tuple[list[str], list[str]]:
    """Convolutions adaptables par LoRA parmi celles correspondant aux motifs.

    Retourne (retenues, ecartees). Une convolution GROUPEE (depthwise, `groups > 1`)
    est ecartee : peft exige alors un rang divisible par `groups`, ce qui imposerait un
    rang de plusieurs dizaines pour un gain nul — une depthwise ne porte qu'une poignee
    de parametres. Les architectures YOLO en contiennent dans le cou, d'ou la necessite
    de ce filtre.
    """
    compiled = [re.compile(p) for p in patterns]
    kept: list[str] = []
    skipped: list[str] = []
    for name, groups in convolutions:
        if not any(c.search(name) for c in compiled):
            continue
        (kept if int(groups) == 1 else skipped).append(name)
    return kept, skipped


def head_index(names: Iterable[str]) -> int:
    """Index du dernier bloc de `model.<i>` : la tete de detection/pose.

    Convention Ultralytics : le reseau est un `Sequential` dont le dernier element est
    la tete. Tout ce qui precede est backbone + cou.
    """
    indices = {int(m.group(1)) for name in names if (m := re.match(r"model\.(\d+)\.", name))}
    if not indices:
        raise ValueError("Aucun module 'model.<i>.' trouve : structure inattendue.")
    return max(indices)


def freeze_patterns_for(names: Iterable[str], trainable: Iterable[str]) -> list[str]:
    """Noms de parametres a geler : tous sauf ceux correspondant a `trainable`."""
    keep = [re.compile(p) for p in trainable]
    return [name for name in names if not any(c.search(name) for c in keep)]


def parameter_report(model: Any) -> dict[str, Any]:
    """Compte des parametres entrainables. A enregistrer dans le manifeste.

    Un ratio inattendu est le seul signal fiable qu'un patch n'a pas pris : sans lui,
    un entrainement "LoRA" ou tout serait entrainable passerait inapercu.
    """
    total = trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": round(trainable / max(total, 1), 6),
        "n_trainable_tensors": len(trainable_names),
        "trainable_sample": trainable_names[:8],
    }


# --- integration Ultralytics ------------------------------------------------
def pose_trainer_class() -> Any:
    """Classe de trainer YOLO-pose de la version installee."""
    from ultralytics.models.yolo.pose import PoseTrainer

    return PoseTrainer


def disable_fuse(model: Any) -> None:
    """Neutralise la fusion conv+BN d'Ultralytics sur un modele patche.

    La fusion suppose une BatchNorm classique par convolution. Elle est donc
    impossible avec une normalisation conditionnelle (elle ecraserait N jeux de
    statistiques en un seul) et fausse avec des adaptateurs non fusionnes (elle
    n'utiliserait que les poids de base). Neutralisee, elle coute un peu de vitesse
    d'inference et ne change aucun resultat.
    """
    import types

    target = getattr(model, "model", model)
    target.fuse = types.MethodType(lambda self, verbose=True: self, target)  # noqa: ARG005


def make_patched_trainer(base_cls: Any, patch: PatchFn | None = None,
                         freeze: PatchFn | None = None,
                         on_batch: Callable[[Any, Any], Any] | None = None,
                         report: dict[str, Any] | None = None,
                         skip_final_eval: bool = False) -> Any:
    """Trainer derive appliquant un patch au modele, puis un gel apres le degel.

    - `patch` s'execute a la construction du modele (`get_model`) ;
    - `freeze` s'execute a la SORTIE de `_setup_train`, donc apres la boucle de degel
      d'Ultralytics, puis a chaque epoque via `_model_train` — le pipeline pouvant etre
      reconstruit en cours de route ;
    - `on_batch` s'execute a chaque lot pretraite, **cote entrainement ET cote
      validation** : le validateur possede son propre `preprocess` et ne passe pas par
      celui du trainer ;
    - `skip_final_eval` desactive l'evaluation finale, qui recharge le meilleur point de
      sauvegarde et le FUSIONNE : sur un modele patche, la fusion echoue ou fausse le
      resultat. Ses metriques ne servent de toute facon qu'au monitoring (§7.1).
    """

    class _PatchedTrainer(base_cls):  # type: ignore[misc, valid-type]
        def get_model(self, cfg: Any = None, weights: Any = None, verbose: bool = True) -> Any:
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if patch is not None:
                patch(model)
                log.info("Patch applique au modele a sa construction.")
            return model

        def _setup_train(self) -> Any:
            result = super()._setup_train()
            # La boucle de degel d'Ultralytics vient de s'executer : notre gel doit venir
            # APRES elle, sinon il est annule sans le moindre effet.
            self._apply_insectpose_freeze()
            return result

        def _apply_insectpose_freeze(self) -> None:
            """Applique le gel et journalise la part reellement entrainable."""
            if freeze is None:
                return
            freeze(self.model)
            summary = parameter_report(self.model)
            log.info("Parametres entrainables : %d / %d (%.2f %%)",
                     summary["trainable_params"], summary["total_params"],
                     100 * summary["trainable_ratio"])
            if report is not None:
                report.update(summary)
            if summary["trainable_params"] == 0:
                raise RuntimeError(
                    "Aucun parametre entrainable apres patch : verifier les motifs de "
                    "selection (l'entrainement ne ferait rien)."
                )

        def _model_train(self) -> Any:
            # Reapplique a chaque epoque : Ultralytics peut reconstruire le pipeline en
            # cours d'entrainement, ce qui relancerait la boucle de degel.
            result = super()._model_train()
            if freeze is not None:
                freeze(self.model)
            return result

        def preprocess_batch(self, batch: Any) -> Any:
            processed = super().preprocess_batch(batch)
            if on_batch is not None:
                on_batch(self, processed)
            return processed

        def get_validator(self) -> Any:
            validator = super().get_validator()
            if on_batch is None:
                return validator
            # Le validateur a son PROPRE preprocess : sans ce relais, le contexte
            # garderait les indices du dernier lot d'entrainement face a un lot de
            # validation de taille differente.
            original = validator.preprocess
            trainer = self

            def _preprocess(batch: Any) -> Any:
                processed = original(batch)
                on_batch(trainer, processed)
                return processed

            validator.preprocess = _preprocess
            return validator

        def final_eval(self) -> Any:
            if skip_final_eval:
                log.info("Evaluation finale d'Ultralytics ignoree (modele patche) : "
                         "ses metriques ne servent qu'au monitoring (§7.1).")
                return None
            return super().final_eval()

    _PatchedTrainer.__name__ = f"Patched{base_cls.__name__}"
    return _PatchedTrainer