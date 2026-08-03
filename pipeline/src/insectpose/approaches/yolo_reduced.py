"""Approche F : YOLO-pose poule prive de certains keypoints (ADR-0027).

Variante de l'approche A ou les pattes et les ailes posterieures sont **retirees des
labels d'entrainement** (`vis = 0`). Question posee : la capacite du reseau, liberee
des points les plus difficiles et les plus mobiles, ameliore-t-elle le positionnement
des autres ?

Precaution de lecture, essentielle : la verite terrain contient toujours ces points, et
l'evaluation les compte. Les metriques `overall` de F sont donc **mecaniquement moins
bonnes** que celles de A et ne sont pas comparables. La comparaison valide se fait sur
les scopes `keypoint:*` des points CONSERVES :

    python scripts/compare_models.py --exclude-keypoints leg hindwing
"""

from __future__ import annotations

from typing import Any

from insectpose.approaches.yolo_pooled import YoloPooledApproach
from insectpose.context import RunContext
from insectpose.data.datamodule import FoldData
from insectpose.data.keypoints import KeypointSchema
from insectpose.registry import register_approach
from insectpose.utils.logging import get_logger

log = get_logger("yolo_reduced")


def dropped_indices(schema: KeypointSchema, patterns: list[str]) -> list[int]:
    """Indices des keypoints dont le nom contient l'un des motifs. Fonction pure."""
    return [i for i, name in enumerate(schema.names)
            if any(str(p).lower() in name.lower() for p in patterns)]


def mask_keypoints(annotations: Any, indices: list[int]) -> Any:
    """Copie des annotations avec `vis = 0` sur les indices donnes.

    Les coordonnees sont conservees telles quelles : c'est la visibilite qui pilote la
    supervision, et un point a vis=0 est masque dans la loss, jamais appris a zero.
    """
    if not indices:
        return annotations
    frame = annotations.copy()
    frame["kpts_vis"] = frame["kpts_vis"].map(
        lambda v: [0 if i in set(indices) else int(x) for i, x in enumerate(v)]
    )
    return frame


@register_approach("yolo_pooled_reduced")
class YoloPooledReducedApproach(YoloPooledApproach):
    """YOLO-pose poule entraine sans supervision sur un sous-ensemble de keypoints."""

    REQUIRED_APPROACH_KEYS = (
        "weights", "max_det", "conf", "iou", "inference_precision", "predict_chunk_size",
        "drop_keypoints",
    )

    def _prepare_data(self, data: FoldData, ctx: RunContext) -> FoldData:
        """Masque les keypoints exclus dans train et val, jamais dans test.

        Le test reste intact : c'est la verite terrain de reference, commune a toutes
        les approches. Masquer aussi le test reviendrait a changer la metrique.
        """
        schema = self._schema(data)
        patterns = [str(p) for p in self.cfg.approach.drop_keypoints]
        indices = dropped_indices(schema, patterns)
        if not indices:
            raise ValueError(
                f"Aucun keypoint ne correspond a {patterns} dans le schema "
                f"'{schema.name}' : l'approche serait identique a yolo_pooled."
            )
        names = [schema.names[i] for i in indices]
        log.info("%d keypoint(s) retire(s) des labels d'entrainement : %s",
                 len(indices), ", ".join(names))
        ctx.extra["dropped_keypoints"] = names
        ctx.extra["n_supervised_keypoints"] = schema.n_keypoints - len(indices)

        from dataclasses import replace as dataclass_replace

        return dataclass_replace(
            data,
            train=dataclass_replace(data.train,
                                    annotations=mask_keypoints(data.train.annotations, indices)),
            val=dataclass_replace(data.val,
                                  annotations=mask_keypoints(data.val.annotations, indices)),
        )


__all__ = ["YoloPooledReducedApproach", "dropped_indices", "mask_keypoints"]