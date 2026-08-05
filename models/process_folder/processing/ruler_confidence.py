"""
ruler_confidence.py
===================

Confiance sur la presence d'une regle dans l'image.

Le probleme de l'ancienne version
---------------------------------
L'ancienne fonction comparait les groupes entre eux : elle repondait a
"QUEL groupe est la regle ?", pas a "Y A-T-IL une regle ?". Quand l'image ne
contient pas de regle, le detecteur trouve souvent un seul groupe parasite,
et l'ancienne regle renvoyait alors 1.0 : la confiance maximale sur le pire cas.

Ici on separe les deux questions :

    confidence = presence x unicite

  * presence : y a-t-il vraiment une regle ? (nouveau, 3 criteres absolus)
  * unicite  : le bon groupe a-t-il ete choisi ? (votre logique d'origine)

Les 3 criteres de presence sont des RAPPORTS : ils ne dependent ni de la
largeur de l'image, ni de son contraste, donc ils sont comparables d'une image
a l'autre.

Cablage dans ruler_detection.py
-------------------------------
Dans detect_ruler_from_gray, remplacer la boucle sur les lignes par :

    from ruler_confidence import row_spectrum, group_evidence, ruler_confidence

    results, n_cycles, row_specs = [], [], []
    for i in row_indices:
        row = reduced_img_gray[i, :]
        rs = row_spectrum(row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE)
        if rs is not None:
            results.append((i, rs.period_px, rs.phase_rad, rs.magnitude))
            n_cycles.append(cycles_observes(row, rs.period_px))
            row_specs.append(rs)

puis, apres le calcul de gid :

    ev = group_evidence(gid, row_specs, rows_arr, periods_arr,
                        reduced_img_gray, n_candidate_rows=len(row_indices))
    conf, debug = ruler_confidence(ev)

row_spectrum expose .period_px / .phase_rad / .magnitude : le reste du pipeline
(trouver_groupes, px_per_mm) est inchange.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
# Seuils : (valeur pivot, largeur de la zone de doute).
# Valeurs de depart plausibles, A RECALIBRER sur vos images -> voir calibrate().
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "snr_db":          (8.0,  2.0),    # nettete du pic
    "phase_coherence": (0.45, 0.12),   # alignement de la regle
    "support_frac":    (0.05, 0.02),   # fraction de lignes concernees
}


# --------------------------------------------------------------------------- #
# Analyse d'une ligne
# --------------------------------------------------------------------------- #
def _noise_floor(mag):
    """Niveau de bruit local du spectre (filtre median le long des frequences).

    Les lignes d'image ont un spectre en 1/f : sans ca, les basses frequences
    ecrasent tout. Diviser par ce plancher rend le pic comparable entre images.
    """
    k = max(11, len(mag) // 16)
    if k % 2 == 0:
        k += 1
    return median_filter(mag, size=k, mode="nearest") + 1e-12


@dataclass
class RowSpectrum:
    """Analyse d'une ligne. Les 3 premiers champs remplacent l'ancien tuple."""
    period_px: float
    phase_rad: float
    magnitude: float     # magnitude brute, pour trouver_groupes (inchange)
    snr_db: float        # pic / bruit local, en dB -> sans dimension
    freq: float          # frequence du pic (cycles/pixel)


def row_spectrum(row, min_period_ratio, max_period_ratio, prominence,
                 whiten=True):
    """FFT d'une ligne -> RowSpectrum, ou None si aucun pic valide.

    Remplace fft_dominant_frequency, en ajoutant snr_db.
    whiten=True cherche le pic sur le spectre debarrasse du fond en 1/f.
    """
    N = len(row)
    if N < 32:
        return None

    w = np.hanning(N)
    x = (row - row.mean()) * w
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N)
    mag = np.abs(X)
    snr = mag / _noise_floor(mag)

    f_min = 1.0 / (max_period_ratio * N)
    f_max = 1.0 / (min_period_ratio * N)
    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() == 0:
        return None

    search = (snr if whiten else mag).copy()
    search[~mask] = 0.0
    if search.max() <= 0:
        return None

    peaks, _ = find_peaks(search, prominence=prominence * search.max())
    if len(peaks) == 0:
        return None

    k = int(peaks[np.argmax(search[peaks])])
    f0 = float(freqs[k])
    if f0 <= 0:
        return None

    return RowSpectrum(
        period_px=1.0 / f0,
        phase_rad=float(np.angle(X[k])),
        magnitude=float(mag[k]),
        snr_db=float(10.0 * np.log10(max(snr[k], 1e-12))),
        freq=f0,
    )


def phase_at(row, f):
    """Phase de la ligne a une frequence donnee (DFT sur un seul point).

    Necessaire car d'une ligne a l'autre le pic peut tomber dans un bin
    different : il faut mesurer toutes les phases du groupe a la MEME frequence,
    sinon l'ecart mesure ne veut rien dire.
    """
    N = len(row)
    x = (row - row.mean()) * np.hanning(N)
    return float(np.angle(x @ np.exp(-2j * np.pi * f * np.arange(N))))


def phase_coherence(rows, phases, n_slopes=1024):
    """Alignement de la regle, dans [0, 1]. C'est le critere le plus fort.

    Une regle inclinee donne une phase qui avance REGULIEREMENT d'une ligne a
    l'autre (droite -> phase constante). Une texture naturelle (herbe, tissu,
    vaguelettes) donne des phases dans le desordre.

    On teste toutes les pentes possibles et on garde la meilleure, puis on
    retranche le score qu'on obtiendrait par pur hasard (sinon on trouve
    toujours une pente qui "marche" avec peu de lignes).
    """
    n = len(rows)
    if n < 6:
        return 0.0
    r = np.asarray(rows, dtype=float)
    r = r - r.mean()
    if r.max() - r.min() <= 0:
        return 0.0

    z = np.exp(1j * np.asarray(phases, dtype=float))
    slopes = np.linspace(-np.pi, np.pi, n_slopes)
    best = float(np.abs(np.exp(-1j * np.outer(slopes, r)) @ z).max() / n)

    hasard = float(np.sqrt(np.log(max(n, 2)) / n))
    if hasard >= 1.0:
        return 0.0
    return float(np.clip((best - hasard) / (1.0 - hasard), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Agregation par groupe
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """Ce qu'on a mesure sur le groupe principal."""
    snr_db: float = 0.0
    phase_coherence: float = 0.0
    support_frac: float = 0.0
    group_snr_db: list = field(default_factory=list)  # [principal, secondaires]
    n_rows: int = 0


def group_evidence(gid, row_specs, rows_arr, periods_arr, reduced_img_gray,
                   n_candidate_rows):
    """Mesure les 3 criteres sur le groupe principal (gid == 0).

    n_candidate_rows : nombre de lignes ANALYSEES (pas seulement retenues).
    """
    ev = Evidence()
    main = np.where(np.asarray(gid) == 0)[0]
    if len(main) == 0:
        return ev

    rows_main = np.asarray(rows_arr, dtype=float)[main]
    periods_main = np.asarray(periods_arr, dtype=float)[main]
    ev.n_rows = len(main)

    # 1. nettete du pic
    ev.snr_db = float(np.median([row_specs[i].snr_db for i in main]))

    # 2. fraction de lignes concernees
    ev.support_frac = len(main) / max(n_candidate_rows, 1)

    # 3. alignement, mesure a une frequence de reference commune au groupe
    f_ref = 1.0 / float(np.median(periods_main))
    phases = np.array([phase_at(reduced_img_gray[int(r), :], f_ref)
                       for r in rows_main])
    ev.phase_coherence = phase_coherence(rows_main, phases)

    # contexte : niveau de chaque groupe, pour l'unicite
    for g in sorted(int(g) for g in np.unique(gid) if g != -1):
        idx = np.where(np.asarray(gid) == g)[0]
        ev.group_snr_db.append(
            float(np.median([row_specs[i].snr_db for i in idx])))
    return ev


# --------------------------------------------------------------------------- #
# Confiance
# --------------------------------------------------------------------------- #
def _score(value, x0, width):
    """Transforme une mesure en note entre 0 et 1 (transition douce en x0)."""
    return float(1.0 / (1.0 + np.exp(-(value - x0) / width)))


def ruler_confidence(ev):
    """Confiance finale dans [0, 1], + un dict de debug a loguer par image.

    presence = moyenne geometrique des 3 notes. Geometrique et non arithmetique :
    une vraie regle doit satisfaire les 3 criteres, donc une note nulle doit
    tout faire tomber, et non etre compensee par les deux autres.
    """
    if ev is None or ev.n_rows == 0:
        return 0.0, {"raison": "aucun groupe"}

    mesures = {
        "snr_db": ev.snr_db,
        "phase_coherence": ev.phase_coherence,
        "support_frac": ev.support_frac,
    }
    notes = {k: max(_score(v, *THRESHOLDS[k]), 1e-3) for k, v in mesures.items()}

    presence = float(np.prod(list(notes.values())) ** (1.0 / len(notes)))
    unicite = _unicite(ev.group_snr_db)
    conf = float(np.clip(presence * unicite, 0.0, 1.0))

    debug = {
        "confidence": conf,
        "presence": presence,
        "unicite": unicite,
        "n_groupes": len(ev.group_snr_db),
        "n_lignes": ev.n_rows,
        "mesures": mesures,
        "notes": notes,
    }
    return conf, debug


def _unicite(group_snr_db):
    """Votre logique d'origine, mais sur le SNR (sans dimension).

    Un seul groupe -> 1.0 est correct ICI : rien ne concurrence la frequence
    retenue. Ce qui etait faux, c'etait d'en deduire qu'une regle existe.
    """
    m = np.asarray(list(group_snr_db), dtype=float)
    if m.size <= 1:
        return 1.0
    if not np.isfinite(m[0]) or m[0] <= 0:
        return 0.0
    return float(np.clip(((m[0] - m[1:]) / m[0]).clip(0, 1).mean(), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Calibration (optionnel mais fortement conseille)
# --------------------------------------------------------------------------- #
def calibrate(evidences_avec_regle, evidences_sans_regle):
    """Propose des seuils a partir d'images dont vous connaissez la reponse.

    Collectez les Evidence de ~30 images avec regle et ~30 sans, passez-les ici,
    et recopiez le resultat dans THRESHOLDS. Si un critere est signale comme
    "chevauchement", c'est qu'il ne separe pas vos images : inutile de le regler.
    """
    propositions = {}
    for name in THRESHOLDS:
        pos = np.array([getattr(e, name) for e in evidences_avec_regle])
        neg = np.array([getattr(e, name) for e in evidences_sans_regle])
        if pos.size == 0 or neg.size == 0:
            continue
        haut_neg = float(np.quantile(neg, 0.90))
        bas_pos = float(np.quantile(pos, 0.10))
        if bas_pos <= haut_neg:
            propositions[name] = (float(np.median([haut_neg, bas_pos])),
                                  THRESHOLDS[name][1], "chevauchement")
        else:
            propositions[name] = (0.5 * (haut_neg + bas_pos),
                                  max((bas_pos - haut_neg) / 4.0, 1e-3), "ok")
    return propositions


# --------------------------------------------------------------------------- #
# Calibration sur annotations (a lancer A LA MAIN, pas dans le pipeline)
# --------------------------------------------------------------------------- #
def _evidence_from_path(img_path, ratio):
    """Rejoue le pipeline complet sur une image -> (Evidence, cycles_median).

    Renvoie (None, 0.0) si aucun groupe n'est trouve (= le detecteur dit
    deja "rien ici", ce qui est une reponse valable qu'aucun seuil ne changera).
    """
    from ruler_detection import (MIN_FREQ_RATIO, MAX_FREQ_RATIO,
                                 PEAK_PROMINENCE, load_image,
                                 cycles_observes, trouver_groupes)

    gray, _ = load_image(str(img_path))
    reduced = gray[::ratio, ::ratio]
    H = reduced.shape[0]

    specs, rows, periods, mags, cycles = [], [], [], [], []
    for i in range(H):
        row = reduced[i, :]
        rs = row_spectrum(row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE)
        if rs is not None:
            specs.append(rs)
            rows.append(i)
            periods.append(rs.period_px)
            mags.append(rs.magnitude)
            cycles.append(cycles_observes(row, rs.period_px))

    if not specs:
        return None, 0.0

    rows = np.asarray(rows, dtype=float)
    periods = np.asarray(periods, dtype=float)
    gid = trouver_groupes(periods, np.asarray(mags), rows, cycles,
                          delta=0.2, n_groupes=5)

    if not np.any(gid == 0):
        return None, 0.0

    ev = group_evidence(gid, specs, rows, periods, reduced, n_candidate_rows=H)
    cyc_main = float(np.median(np.asarray(cycles, dtype=float)[gid == 0]))
    return ev, cyc_main


def calibrate_from_annotations(json_path, images_root=None, ratio=5,
                               positive_label=2, verbose=True):
    """Calcule les seuils optimaux a partir d'annotation.json. USAGE MANUEL.

    annotation.json : {"chemin/image.jpg": "2", ...}
        0 = rien, 1 = scale bar, 2 = regle.

    La classe POSITIVE est la classe 2 (regle). Les classes 0 ET 1 sont des
    negatifs : une scale bar ne doit pas etre lue comme une regle. Le rapport
    donne l'AUC separement contre chaque classe, pour voir laquelle resiste.

    Retourne (thresholds, rapport) :
      * thresholds : dict pret a recopier dans THRESHOLDS
      * rapport    : diagnostics par critere + performance de la confiance finale

    Exemple :
        th, rap = calibrate_from_annotations("annotation.json", images_root="data")
        print(rap["decision"])
    """
    import json
    from pathlib import Path

    from scipy.stats import rankdata

    root = Path(images_root) if images_root else None
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # --- 1. extraction des features -----------------------------------------
    data = {0: [], 1: [], 2: []}      # label -> liste de (Evidence, cycles)
    n_sans_groupe = {0: 0, 1: 0, 2: 0}
    n_illisibles = 0

    for rel_path, label in annotations.items():
        try:
            label = int(label)
        except (TypeError, ValueError):
            continue
        if label not in data:
            continue

        path = root / rel_path if root else Path(rel_path)
        try:
            ev, cyc = _evidence_from_path(path, ratio)
        except Exception as err:                      # image absente/corrompue
            n_illisibles += 1
            if verbose:
                print(f"  ignoree ({type(err).__name__}) : {rel_path}")
            continue

        if ev is None:
            n_sans_groupe[label] += 1
        else:
            data[label].append((ev, cyc))

        if verbose and (sum(len(v) for v in data.values()) % 25 == 0):
            print(f"  ... {sum(len(v) for v in data.values())} images traitees")

    pos = [e for e, _ in data[positive_label]]
    neg_par_classe = {lab: [e for e, _ in data[lab]]
                      for lab in data if lab != positive_label}
    neg = [e for lst in neg_par_classe.values() for e in lst]

    if len(pos) < 5 or len(neg) < 5:
        raise ValueError(
            f"Pas assez d'images exploitables : {len(pos)} positives, "
            f"{len(neg)} negatives. Verifiez les chemins du json.")

    # --- 2. outils de separation --------------------------------------------
    def auc(a, b):
        """Aire sous la courbe ROC (Mann-Whitney). 0.5 = inutile, 1 = parfait."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return float("nan")
        r = rankdata(np.concatenate([a, b]))
        return float((r[:a.size].sum() - a.size * (a.size + 1) / 2)
                     / (a.size * b.size))

    def meilleur_seuil(a, b):
        """Seuil maximisant (taux de vrais positifs - taux de faux positifs)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        vals = np.unique(np.concatenate([a, b]))
        if vals.size < 2:
            return float(vals[0]) if vals.size else 0.0, 0.0
        cands = (vals[:-1] + vals[1:]) / 2.0
        j = (a[:, None] >= cands).mean(0) - (b[:, None] >= cands).mean(0)
        k = int(np.argmax(j))
        return float(cands[k]), float(j[k])

    # --- 3. un seuil par critere --------------------------------------------
    thresholds, rapport_criteres = {}, {}
    for name in THRESHOLDS:
        v_pos = np.array([getattr(e, name) for e in pos])
        v_neg = np.array([getattr(e, name) for e in neg])

        x0, j = meilleur_seuil(v_pos, v_neg)

        # largeur = etendue de la zone de doute entre les deux classes.
        # classes bien separees -> ecart positif  -> transition nette
        # classes qui se chevauchent -> ecart negatif -> transition plus douce
        ecart = abs(float(np.quantile(v_pos, 0.10) - np.quantile(v_neg, 0.90)))
        etendue = float(np.ptp(np.concatenate([v_pos, v_neg])))
        width = max(ecart / 4.0, 0.05 * etendue, 1e-3)

        thresholds[name] = (round(x0, 4), round(width, 4))
        rapport_criteres[name] = {
            "x0": round(x0, 4),
            "width": round(width, 4),
            "youden_J": round(j, 3),
            "auc_vs_tout": round(auc(v_pos, v_neg), 3),
            "auc_vs_rien": round(
                auc(v_pos, [getattr(e, name) for e in neg_par_classe.get(0, [])]), 3),
            "auc_vs_scalebar": round(
                auc(v_pos, [getattr(e, name) for e in neg_par_classe.get(1, [])]), 3),
            "mediane_regle": round(float(np.median(v_pos)), 3),
            "mediane_scalebar": round(float(np.median(
                [getattr(e, name) for e in neg_par_classe.get(1, [])] or [np.nan])), 3),
            "mediane_rien": round(float(np.median(
                [getattr(e, name) for e in neg_par_classe.get(0, [])] or [np.nan])), 3),
        }

    # --- 4. critere candidat non retenu : le nombre de graduations ----------
    # C'est lui qui separe le mieux une regle (beaucoup de graduations) d'une
    # scale bar (quelques traits). S'il ressort meilleur que les 3 criteres
    # actuels contre la classe 1, il faut le rajouter dans THRESHOLDS.
    cyc = {lab: np.array([c for _, c in data[lab]]) for lab in data}
    candidat_cycles = {
        "auc_vs_scalebar": round(auc(cyc[positive_label], cyc.get(1, [])), 3),
        "auc_vs_rien": round(auc(cyc[positive_label], cyc.get(0, [])), 3),
        "mediane_regle": round(float(np.median(cyc[positive_label])), 1),
        "mediane_scalebar": round(float(np.median(cyc[1])), 1) if len(cyc[1]) else None,
    }

    # --- 5. performance de la confiance finale avec ces seuils ---------------
    def conf_avec(ev, th):
        notes = [max(_score(getattr(ev, n), *th[n]), 1e-3) for n in th]
        presence = float(np.prod(notes) ** (1.0 / len(notes)))
        return presence * _unicite(ev.group_snr_db)

    c_pos = np.array([conf_avec(e, thresholds) for e in pos])
    c_neg = np.array([conf_avec(e, thresholds) for e in neg])
    seuil_decision, _ = meilleur_seuil(c_pos, c_neg)

    # les images sans groupe comptent comme confiance 0
    n_pos_tot = len(pos) + n_sans_groupe[positive_label]
    n_neg_tot = len(neg) + sum(n_sans_groupe[l] for l in n_sans_groupe
                               if l != positive_label)
    rappel = float((c_pos >= seuil_decision).sum()) / max(n_pos_tot, 1)
    faux_pos = float((c_neg >= seuil_decision).sum()) / max(n_neg_tot, 1)

    rapport = {
        "n_images": {"regle": n_pos_tot, "scalebar": len(neg_par_classe.get(1, [])),
                     "rien": len(neg_par_classe.get(0, [])),
                     "illisibles": n_illisibles},
        "sans_groupe": n_sans_groupe,
        "criteres": rapport_criteres,
        "candidat_nb_graduations": candidat_cycles,
        "decision": {
            "seuil_confidence": round(seuil_decision, 3),
            "rappel_regles": round(rappel, 3),
            "taux_faux_positifs": round(faux_pos, 3),
            "auc_confidence": round(auc(c_pos, c_neg), 3),
        },
    }

    if verbose:
        print("\n--- seuils proposes (a recopier dans THRESHOLDS) ---")
        for k, v in thresholds.items():
            print(f'    "{k}": {v},   # AUC vs scalebar = '
                  f'{rapport_criteres[k]["auc_vs_scalebar"]}')
        print(f"\nseuil de decision sur la confidence : "
              f"{rapport['decision']['seuil_confidence']}")
        print(f"rappel {rapport['decision']['rappel_regles']}  |  "
              f"faux positifs {rapport['decision']['taux_faux_positifs']}")

    return thresholds, rapport

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Calibrage de la confiance sur la presence d'une regle")
    parser.add_argument("json", help="annotation.json")
    parser.add_argument("--images_root", help="racine des images (optionnel)")
    parser.add_argument("--ratio", type=int, default=5,
                        help="reduction de l'image pour le calcul FFT")
    args = parser.parse_args()

    calibrate_from_annotations(args.json, images_root=args.images_root,
                               ratio=args.ratio, verbose=True)