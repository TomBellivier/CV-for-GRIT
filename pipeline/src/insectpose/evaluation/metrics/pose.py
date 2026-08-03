"""Metriques de pose : OKS-AP/AR, PCK, NME, taux de detection par keypoint.

Les instances GT non appariees comptent comme echec quand
`eval.count_missed_gt_as_failure` est vrai : une pipeline qui ne detecte pas
l'insecte n'a pas "0 keypoint evalue", elle a un echec (§7.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from insectpose.data.keypoints import KeypointSchema
from insectpose.evaluation.bundle import EvalBundle, record
from insectpose.evaluation.matching import assign_greedy, average_precision
from insectpose.registry import register_metric
from insectpose.utils.geometry import bbox_diag


def compute_normalizer(spec: Any, schema: KeypointSchema, gt_kpts: np.ndarray,
                       gt_vis: np.ndarray, gt_bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Echelle de reference de chaque instance pour le PCK et la NME (ADR-0009).

    Retourne (valeurs, masque de repli). Le repli est COMPTE et publie : une echelle
    de reference silencieusement remplacee fausserait la comparaison entre approches.
    """
    fallback = bbox_diag(gt_bbox)
    kind = str(spec.type)
    if kind == "bbox_diag":
        return fallback, np.zeros(len(gt_bbox), dtype=bool)
    if kind != "keypoint_distance":
        raise ValueError(f"pck.normalizer.type inconnu : '{kind}'.")

    idx = [schema.index(str(n)) for n in spec.keypoints]
    usable = gt_vis[:, idx].all(axis=1)
    values = np.linalg.norm(gt_kpts[:, idx[0], :] - gt_kpts[:, idx[1], :], axis=-1)
    usable &= values > 1e-6
    if str(spec.get("fallback", "bbox_diag")) == "none":
        return np.where(usable, values, np.nan), ~usable
    return np.where(usable, values, fallback), ~usable


@register_metric("oks_ap_ar")
def oks_ap_ar(bundle: EvalBundle) -> list[dict[str, Any]]:
    """OKS-AP (moyenne sur les seuils) et OKS-AR, metrique primaire par defaut."""
    thresholds = [float(t) for t in bundle.cfg.oks.thresholds]
    out: list[dict[str, Any]] = []
    for scope, pairs in bundle.scopes():
        n_gt = bundle.n_gt(pairs)
        if n_gt == 0:
            continue
        aps, ars = [], []
        for t in thresholds:
            scores, tps, matched_gt_total = [], [], 0
            for p in pairs:
                matched, _ = assign_greedy(p.oks, p.scores, t)
                scores.append(p.scores)
                tps.append(matched >= 0)
                matched_gt_total += int((matched >= 0).sum())
            aps.append(
                average_precision(
                    np.concatenate(scores) if scores else np.zeros(0),
                    np.concatenate(tps) if tps else np.zeros(0, dtype=bool),
                    n_gt,
                )
            )
            ars.append(matched_gt_total / n_gt)
        out.append(record(scope, "oks_ap", float(np.nanmean(aps)), n_gt))
        out.append(record(scope, "oks_ap@0.5", float(aps[0]), n_gt))
        out.append(record(scope, "oks_ar", float(np.nanmean(ars)), n_gt))
    return out


@dataclass
class _Block:
    """Distances normalisees d'un groupe d'instances partageant le MEME schema.

    Les schemas de keypoints n'ont pas le meme K d'un dataset a l'autre : les
    metriques ponctuelles agregent donc des compteurs, jamais des tableaux empiles.
    """

    schema: str
    dataset: str
    dist: np.ndarray      # (N_gt, K) - +inf pour une instance GT non appariee
    valid: np.ndarray     # (N_gt, K) - keypoints annotes (vis > 0)
    fallback: np.ndarray  # (N_gt,) - echelle de reference remplacee par le repli
    conf: np.ndarray      # (N_gt, K) - confiance predite, NaN si non appariee


def _pointwise(bundle: EvalBundle, pairs: list) -> list[_Block]:
    """Distances normalisees par keypoint, groupees par schema.

    Une GT non appariee garde dist = +inf : elle echoue tous les seuils PCK au lieu
    de disparaitre du denominateur (§7.2).
    """
    thr = float(bundle.cfg.match_oks_threshold)
    score_thr = float(bundle.cfg.score_threshold_pointwise)
    acc: dict[tuple[str, str], list[tuple[np.ndarray, ...]]] = {}

    for p in pairs:
        if p.n_gt == 0:
            continue
        schema_name = str(bundle.gt.loc[p.gt_rows[0], "keypoint_schema"])
        gt_kpts = bundle.gt_array("kpts_xy", p.gt_rows).reshape(p.n_gt, -1, 2)
        gt_vis = np.stack(
            bundle.gt.loc[p.gt_rows, "kpts_vis"].map(lambda v: np.asarray(v, int)).to_numpy()
        )
        gt_bbox = bundle.gt_array("bbox_xywh", p.gt_rows)
        norm, fell_back = compute_normalizer(
            bundle.cfg.pck.normalizer, bundle.schemas[schema_name], gt_kpts, gt_vis > 0, gt_bbox
        )
        d = np.full((p.n_gt, gt_kpts.shape[1]), np.inf, dtype=float)
        conf = np.full_like(d, np.nan)

        if p.n_pred:
            keep = p.scores >= score_thr
            sim = np.where(keep[:, None], p.oks, -1.0)
            matched, _ = assign_greedy(sim, p.scores, thr)
            pred_kpts = bundle.pred_array("kpts_xy", p.pred_rows).reshape(p.n_pred, -1, 2)
            pred_conf = bundle.pred_array("kpts_score", p.pred_rows)
            for pi, gi in enumerate(matched):
                if gi >= 0:
                    d[gi] = np.linalg.norm(pred_kpts[pi] - gt_kpts[gi], axis=-1) / max(
                        norm[gi], 1e-9
                    )
                    conf[gi] = pred_conf[pi]
        acc.setdefault((schema_name, p.dataset), []).append((d, gt_vis > 0, fell_back, conf))

    return [
        _Block(schema=schema, dataset=dataset,
               dist=np.concatenate([b[0] for b in blocks]),
               valid=np.concatenate([b[1] for b in blocks]),
               fallback=np.concatenate([b[2] for b in blocks]),
               conf=np.concatenate([b[3] for b in blocks]))
        for (schema, dataset), blocks in acc.items()
    ]


@register_metric("pck")
def pck(bundle: EvalBundle) -> list[dict[str, Any]]:
    """PCK@alpha normalise (par defaut diagonale de bbox GT ; DECISION OPEN-02)."""
    normalizer = str(bundle.cfg.pck.normalizer.name)
    out: list[dict[str, Any]] = []
    for scope, pairs in bundle.scopes():
        blocks = _pointwise(bundle, pairs)
        if not blocks:
            continue
        for alpha in [float(a) for a in bundle.cfg.pck.alphas]:
            ok = sum(int(((b.dist <= alpha) & b.valid).sum()) for b in blocks)
            total = sum(int(b.valid.sum()) for b in blocks)
            if total == 0:
                continue
            out.append(record(scope, f"pck@{alpha:g}_{normalizer}", ok / total, total))
        n_inst = sum(len(b.fallback) for b in blocks)
        n_fallback = sum(int(b.fallback.sum()) for b in blocks)
        if n_inst:
            out.append(record(scope, "pck_normalizer_fallback_rate", n_fallback / n_inst, n_inst))
    return out


@register_metric("nme")
def nme(bundle: EvalBundle) -> list[dict[str, Any]]:
    """Erreur moyenne normalisee. Deux variantes, jamais confondues."""
    out: list[dict[str, Any]] = []
    for scope, pairs in bundle.scopes():
        blocks = _pointwise(bundle, pairs)
        if not blocks:
            continue
        total = sum(int(b.valid.sum()) for b in blocks)
        if total == 0:
            continue
        finite_sum, finite_n = 0.0, 0
        for b in blocks:
            finite = np.isfinite(b.dist) & b.valid
            finite_sum += float(b.dist[finite].sum())
            finite_n += int(finite.sum())
        if finite_n:
            # variante 'appariee' : ne dit rien des instances manquees
            out.append(record(scope, "nme_matched_only", finite_sum / finite_n, finite_n))
        out.append(record(scope, "kpt_coverage", finite_n / total, total))
    return out


@register_metric("keypoint_rate")
def keypoint_rate(bundle: EvalBundle) -> list[dict[str, Any]]:
    """Detail par keypoint : PCK, erreur normalisee et confiance predite.

    La confiance et l'erreur sont publiees par point car leur relation dit si la
    confiance du modele est exploitable comme filtre en production (§8.3).
    """
    if not bool(bundle.cfg.scopes.per_keypoint):
        return []
    alpha = float(bundle.cfg.pck.reference_alpha)
    normalizer = str(bundle.cfg.pck.normalizer.name)
    out: list[dict[str, Any]] = []
    for block in _pointwise(bundle, bundle.pairs):
        schema = bundle.schemas[block.schema]
        for k, name in enumerate(schema.names):
            v = block.valid[:, k]
            if not v.any():
                continue
            scope = f"keypoint:{block.dataset}:{name}"
            n = int(v.sum())
            dist = block.dist[v, k]
            out.append(record(scope, f"pck@{alpha:g}_{normalizer}",
                              float((dist <= alpha).sum()) / n, n))
            finite = np.isfinite(dist)
            if finite.any():
                out.append(record(scope, "nme", float(dist[finite].mean()), int(finite.sum())))
            conf = block.conf[v, k]
            conf = conf[np.isfinite(conf)]
            if conf.size:
                out.append(record(scope, "kpt_conf_mean", float(conf.mean()), int(conf.size)))
    return out