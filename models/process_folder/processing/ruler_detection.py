"""
ruler_detection.py
==================

Adapted from your original file. The Fourier logic is unchanged; the edits are:

  1. detect_ruler now ALSO returns the per-group mean FFT magnitudes, ordered
     [main_group, secondary_1, ...]. confidence.ruler_confidence() turns that
     list into the ruler confidence you specified (single group -> 1, several
     groups -> mean relative magnitude gap to the main group).
     New return signature:  (px_per_mm, line, group_magnitudes)

  2. Bug fix: fft_dominant_frequency returned only 2 values (None, None) on
     failure while the caller unpacks 3 -> it now returns (None, None, None).

  3. Bug fix: the "no group discovered" test now counts the REAL groups (it
     excludes the -1 "unclassified" label) instead of relying on the length of
     np.unique(gid).

The in-code comments of the original (in French) are kept where they document
the algorithm.
"""

import warnings

import numpy as np
from PIL import Image
from scipy.signal import find_peaks, hilbert

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Parameters (kept from the original; the pipeline passes ratio/graduation in)
# --------------------------------------------------------------------------- #
GRADUATION_MM = 1.0

MIN_FREQ_RATIO = 0.005
MAX_FREQ_RATIO = 0.1
PEAK_PROMINENCE = 0.1

RATIO = 5

N_LINES = None


def load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(path).convert("RGB")
    img_color = np.array(img)
    img_gray = np.dot(img_color[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0
    return img_gray, img_color


def fft_dominant_frequency(row: np.ndarray,
                           min_period_ratio: float,
                           max_period_ratio: float,
                           prominence: float):
    """FFT of one pixel row -> (dominant period px, phase rad, peak magnitude).

    Returns (None, None, None) if no valid peak is found.
    """
    N = len(row)
    row_centered = row - row.mean()
    window = np.hanning(N)
    row_windowed = row_centered * window

    fft_vals = np.fft.rfft(row_windowed)
    freqs = np.fft.rfftfreq(N)
    magnitude = np.abs(fft_vals)

    f_min = 1.0 / (max_period_ratio * N)
    f_max = 1.0 / (min_period_ratio * N)

    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() == 0:
        return None, None, None            # bug fix: 3 values

    mag_masked = magnitude.copy()
    mag_masked[~mask] = 0

    peaks, props = find_peaks(mag_masked, prominence=prominence * mag_masked.max())
    if len(peaks) == 0:
        return None, None, None            # bug fix: 3 values

    best_peak = peaks[np.argmax(mag_masked[peaks])]
    period_px = 1.0 / freqs[best_peak]
    phase_rad = np.angle(fft_vals[best_peak])
    return period_px, phase_rad, max(mag_masked[peaks])


def cycles_observes(row, period_px, seuil_env=0.2):
    N = len(row)
    f0 = 1.0 / period_px
    row_c = row - row.mean()                 # PAS de fenetre ici (on garde les bords)

    fft = np.fft.rfft(row_c)
    freqs = np.fft.rfftfreq(N)
    bw = 0.5 * f0                            # +/-50 % autour de f0
    fft[(freqs < f0 - bw) | (freqs > f0 + bw)] = 0
    filtre = np.fft.irfft(fft, n=N)

    env = np.abs(hilbert(filtre))            # enveloppe d'amplitude
    seuil = seuil_env * env.max()

    peaks, _ = find_peaks(filtre, distance=max(1, period_px * 0.5))
    peaks = peaks[env[peaks] >= seuil]
    return len(peaks)


def _pente_initiale(graine, period, lines, ordre, rang, delta, win=3):
    """Estime la pente locale period=f(lines) autour de la graine."""
    pos = rang[graine]
    fenetre = 4.0 * delta
    idx = []
    for d in range(-win, win + 1):
        p = pos + d
        if 0 <= p < len(ordre):
            c = ordre[p]
            if abs(period[c] - period[graine]) <= fenetre:
                idx.append(c)
    if len(idx) >= 2:
        a, _ = np.polyfit(lines[idx], period[idx], 1)
        return a
    return 0.0


def _croissance(graine, direction, membres, groupe_id, gid,
                period, lines, n_cycles, ordre, rang, delta, max_sauts, pente0):
    """Etend le groupe depuis la graine dans un sens de l'axe 'lines'."""
    pos = rang[graine]
    sauts = 0
    n = len(ordre)
    while True:
        pos += direction
        if pos < 0 or pos >= n:
            break
        cand = ordre[pos]

        if groupe_id[cand] != -1:            # candidat deja attribue -> saut
            sauts += 1
            if sauts > max_sauts:
                break
            continue

        lm = lines[membres]
        pm = period[membres]
        cm = n_cycles[membres]
        if len(membres) >= 2:
            a, b = np.polyfit(lm, pm, 1)
            period_pred = a * lines[cand] + b
        else:
            period_pred = pm[0] + pente0 * (lines[cand] - lm[0])

        if len(membres) >= 2:
            a, b = np.polyfit(lm, cm, 1)
            cycles_pred = a * lines[cand] + b
        else:
            cycles_pred = cm[0] + pente0 * (lines[cand] - lm[0])

        if abs(period[cand] - period_pred) <= delta:
            groupe_id[cand] = gid
            membres.append(cand)
            sauts = 0
        else:
            sauts += 1
            if sauts > max_sauts:
                break


def trouver_groupes(period, magnitude, lines, n_cycles, delta,
                    max_sauts=2, n_groupes=None, min_cycles=20):
    """Regroupe les points (period vs lines) alignes sur une meme droite.

    Retour: groupe_id (int array). Group 0 = groupe le plus peuple apres
    re-etiquetage.
    """
    period = np.asarray(period, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    lines = np.asarray(lines, dtype=float)
    n_cycles = np.asarray(n_cycles, dtype=float)
    n = len(period)

    groupe_id = np.full(n, -1, dtype=int)    # -1 = libre, -2 = rejete
    ordre = np.argsort(lines)
    rang = np.empty(n, dtype=int)
    rang[ordre] = np.arange(n)

    gid = 0
    while np.any(groupe_id == -1):
        if n_groupes is not None and gid >= n_groupes:
            break
        libres = np.where(groupe_id == -1)[0]
        graine = libres[np.argmax(magnitude[libres])]

        membres = [graine]
        groupe_id[graine] = gid
        pente0 = _pente_initiale(graine, period, lines, ordre, rang, delta)
        _croissance(graine, +1, membres, groupe_id, gid,
                    period, lines, n_cycles, ordre, rang, delta, max_sauts, pente0)
        _croissance(graine, -1, membres, groupe_id, gid,
                    period, lines, n_cycles, ordre, rang, delta, max_sauts, pente0)

        if n_cycles[membres].mean() > min_cycles:
            gid += 1
        else:
            groupe_id[membres] = -2

    groupe_id[groupe_id == -2] = -1

    # re-etiquetage : groupe 0 = le plus peuple
    ids = [g for g in np.unique(groupe_id) if g != -1]
    moyennes = {g: len(np.where(groupe_id == g)[0]) for g in ids}
    ordre_groupes = sorted(ids, key=lambda g: moyennes[g], reverse=True)
    remap = {ancien: nouveau for nouveau, ancien in enumerate(ordre_groupes)}

    nouveau_id = np.full_like(groupe_id, -1)
    for ancien, nouveau in remap.items():
        nouveau_id[groupe_id == ancien] = nouveau
    return nouveau_id


def gray_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Luma of an RGB uint8 image, matching load_image (R,G,B weights)."""
    return np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0


def detect_ruler(img_path, ratio=RATIO):
    """Path-based wrapper (kept for backward compatibility)."""
    base_img_gray, _base_img_color = load_image(img_path)
    return detect_ruler_from_gray(base_img_gray, ratio=ratio)


def detect_ruler_from_rgb(img_rgb: np.ndarray, ratio=RATIO):
    """Array-based entry point (RGB image already in memory)."""
    return detect_ruler_from_gray(gray_from_rgb(img_rgb), ratio=ratio)


def detect_ruler_from_gray(base_img_gray: np.ndarray, ratio=RATIO):
    """Detect a ruler and return (px_per_mm, line, group_magnitudes).

    group_magnitudes : list of the groups' mean FFT magnitudes, ordered
                       [main_group, secondary_1, ...]. Passed to
                       confidence.ruler_confidence(). Returns (None, None, None)
                       when no ruler frequency is found.

    This is the shared core: detect_ruler (path) and detect_ruler_from_rgb
    (in-memory image, used for Hugging Face) both funnel through here.
    """
    reduced_img_gray = base_img_gray[::ratio, ::ratio]
    H, _W = reduced_img_gray.shape

    row_indices = np.arange(H)
    if N_LINES is not None and N_LINES < H:
        row_indices = np.linspace(0, H - 1, N_LINES, dtype=int)

    results = []
    n_cycles = []
    for i in row_indices:
        row = reduced_img_gray[i, :]
        period, phase, magnitude = fft_dominant_frequency(
            row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE
        )
        if period is not None:
            results.append((i, period, phase, magnitude))
            n_cycles.append(cycles_observes(row, period))

    # No row produced a valid dominant frequency -> no ruler.
    if len(results) == 0:
        return None, None, None

    rows_arr = np.array([r[0] for r in results], dtype=float)
    phases_arr = np.array([r[2] for r in results], dtype=float)
    mag_arr = np.array([r[3] for r in results], dtype=float)
    periods_arr = np.array([r[1] for r in results], dtype=float)

    gid = trouver_groupes(periods_arr, mag_arr, rows_arr, n_cycles,
                          delta=0.2, n_groupes=5)

    # Real groups only (exclude the -1 "unclassified" label).
    real_groups = [g for g in np.unique(gid) if g != -1]
    if len(real_groups) == 0:
        return None, None, None

    # Main group (label 0) -> px/mm.
    indices = np.where(gid == 0)[0]
    max_line_idx = int(indices.mean())
    mean_period = np.asarray(periods_arr)[indices].mean()

    T_median = mean_period * ratio
    px_per_mm = T_median / GRADUATION_MM

    # Per-group mean magnitude, ordered [main=0, secondary=1, 2, ...].
    group_magnitudes = [float(mag_arr[gid == g].mean()) for g in sorted(real_groups)]

    return px_per_mm, max_line_idx * ratio, group_magnitudes
