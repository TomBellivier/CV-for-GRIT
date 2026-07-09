import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks, hilbert
from scipy.stats import circmean
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
                            prominence: float) -> tuple[float | None, float | None]:
    """
    Applique la FFT sur une ligne de pixels et retourne
    la période dominante (en px) et sa phase.

    Paramètres
    ----------
    row           : 1D array de niveaux de gris
    min_period_px : période minimale acceptée (px)
    max_period_px : période maximale acceptée (px)
    prominence    : proéminence relative minimale du pic

    Retourne
    --------
    (période_px, phase_rad) ou (None, None) si aucun pic valide
    """
    N = len(row)

    row_centered = row - row.mean()

    window = np.hanning(N)
    row_windowed = row_centered * window

    fft_vals = np.fft.rfft(row_windowed)
    freqs    = np.fft.rfftfreq(N)   # fréquences normalisées [0, 0.5]

    magnitude = np.abs(fft_vals)

    f_min = 1.0 / (max_period_ratio * N)
    f_max = 1.0 / (min_period_ratio * N)

    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() == 0:
        return None, None

    mag_masked = magnitude.copy()
    mag_masked[~mask] = 0

    peaks, props = find_peaks(mag_masked,
                              prominence=prominence * mag_masked.max())
    if len(peaks) == 0:
        return None, None

    best_peak = peaks[np.argmax(mag_masked[peaks])]

    period_px = 1.0 / freqs[best_peak]
    phase_rad = np.angle(fft_vals[best_peak])

    return period_px, phase_rad, max(mag_masked[peaks])

def cycles_observes(row, period_px, seuil_env=0.3):
    N = len(row)
    f0 = 1.0 / period_px
    row_c = row - row.mean()                 # PAS de fenêtre ici (on garde les bords)

    # bande passante autour de f0
    fft   = np.fft.rfft(row_c)
    freqs = np.fft.rfftfreq(N)
    bw    = 0.5 * f0                          # ±50 % autour de f0
    fft[(freqs < f0 - bw) | (freqs > f0 + bw)] = 0
    filtre = np.fft.irfft(fft, n=N)

    # enveloppe d'amplitude de la composante
    env = np.abs(hilbert(filtre))
    seuil = seuil_env * env.max()

    # crêtes de l'oscillation, uniquement là où elle est réellement présente
    peaks, _ = find_peaks(filtre, distance=max(1, period_px * 0.5))
    peaks = peaks[env[peaks] >= seuil]
    return len(peaks)

def _pente_initiale(graine, period, lines, ordre, rang, delta, win=3):
    """Estime la pente locale period=f(lines) autour de la graine, à partir
    des points voisins (sur l'axe 'lines') restant proches en period.
    Utile quand la crête est inclinée et que la graine n'a qu'un voisin."""
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
    """Étend le groupe depuis la graine dans un sens de l'axe 'lines'."""
    pos = rang[graine]
    sauts = 0
    n = len(ordre)
    while True:
        pos += direction
        if pos < 0 or pos >= n:
            break
        cand = ordre[pos]

        # candidat déjà attribué -> on saute (trous tolérés jusqu'à max_sauts)
        if groupe_id[cand] != -1:
            sauts += 1
            if sauts > max_sauts:
                break
            continue

        # period attendu sur la droite du groupe, à cette valeur de 'lines'
        lm = lines[membres]
        pm = period[membres]
        cm = n_cycles[membres]
        if len(membres) >= 2:
            a, b = np.polyfit(lm, pm, 1)         # ajustement du groupe
            period_pred = a * lines[cand] + b
        else:
            period_pred = pm[0] + pente0 * (lines[cand] - lm[0])  # graine seule

        if len(membres) >= 2:
            a, b = np.polyfit(lm, cm, 1)         # ajustement du groupe
            cycles_pred = a * lines[cand] + b
        else:
            cycles_pred = cm[0] + pente0 * (lines[cand] - lm[0])  # graine seule

        if abs(period[cand] - period_pred) <= delta and abs(n_cycles[cand] - cycles_pred) <= delta:
            groupe_id[cand] = gid
            membres.append(cand)
            sauts = 0
        else:
            sauts += 1
            if sauts > max_sauts:
                break

def trouver_groupes(period, magnitude, lines, n_cycles, delta, max_sauts=2, n_groupes=None, min_cycles=20):
    """
    Regroupe les points (period vs lines) alignés sur une même droite.

    Paramètres
    ----------
    period, magnitude, lines : array-like (même taille)
    delta : float
        Tolérance en 'period' autour de la droite du groupe.
    max_sauts : int
        Nombre de 'lines' manquantes/rejetées tolérées avant d'arrêter
        l'extension d'un groupe (gère les trous dans une crête).

    Retour
    ------
    groupe_id : np.ndarray d'entiers (même taille)
        Identifiant de groupe (>= 0) de chaque point. La graine d'un groupe
        (son point de magnitude max) est garantie comme étant le max du groupe.
    """
    period    = np.asarray(period, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    lines     = np.asarray(lines, dtype=float)
    n_cycles   = np.asarray(n_cycles, dtype=float)
    n = len(period)

    groupe_id = np.full(n, -1, dtype=int)        # -1 = libre, -2 = rejeté
    ordre = np.argsort(lines)
    rang = np.empty(n, dtype=int); rang[ordre] = np.arange(n)

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

        # validation : moyenne des cycles du groupe > min_cycles
        if n_cycles[membres].mean() > min_cycles:
            gid += 1                              # groupe validé
        else:
            groupe_id[membres] = -2               # rejeté, exclu des graines

    groupe_id[groupe_id == -2] = -1               # les rejetés -> non classés
        
     # --- ré-étiquetage : groupe 0 = meilleure magnitude moyenne par point ---
    ids = [g for g in np.unique(groupe_id) if g != -1]
    moyennes = {g: magnitude[groupe_id == g].mean() for g in ids}
    ordre_groupes = sorted(ids, key=lambda g: moyennes[g], reverse=True)
    remap = {ancien: nouveau for nouveau, ancien in enumerate(ordre_groupes)}

    nouveau_id = np.full_like(groupe_id, -1)
    for ancien, nouveau in remap.items():
        nouveau_id[groupe_id == ancien] = nouveau
    groupe_id = nouveau_id

    return groupe_id

def detect_ruler(img_path, ratio=RATIO):
    base_img_gray, base_img_color = load_image(img_path)
    reduced_img_gray = base_img_gray[::ratio, ::ratio]
    H, W = reduced_img_gray.shape

    row_indices = np.arange(H)
    if N_LINES is not None and N_LINES < H:
        row_indices = np.linspace(0, H - 1, N_LINES, dtype=int)

    results = []   
    n_cycles = []

    for i in row_indices:
        # row = img_gray[i, :]
        row = reduced_img_gray[i, :]
        period, phase, magnitude = fft_dominant_frequency(
            row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE
        )
        if period is not None:
            results.append((i, period, phase, magnitude))
            n_cycles.append(cycles_observes(row, period))
    
    rows_arr   = np.array([r[0] for r in results], dtype=float)
    phases_arr  = np.array([r[2] for r in results], dtype=float)
    mag_arr = np.array([r[3] for r in results], dtype=float)

    periods_arr = np.array([r[1] for r in results], dtype=float)

    period, magnitude, lines = periods_arr, mag_arr, rows_arr

    gid = trouver_groupes(period, magnitude, lines, n_cycles, delta=0.2, n_groupes=5)

    if len(np.unique(gid)) < 2: # no group discovered
        return None, None
    indices = np.where(gid == 0)[0]
    max_line_idx = int(indices.mean())
    mean_period = np.asarray(periods_arr)[indices].mean()
    mean_phase = np.asarray(phases_arr)[indices].mean()

    T_median = mean_period * ratio
    px_per_mm = T_median / GRADUATION_MM 

    return px_per_mm, max_line_idx * ratio
