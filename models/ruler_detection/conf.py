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