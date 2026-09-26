"""
ruler_detection.py
==================

Adapted from the original file. The Fourier logic is unchanged; the edits are:

  1. detect_ruler now returns a ruler confidence computed by
     ruler_confidence.ruler_confidence() (presence x uniqueness of the main group).
     Return signature:  (px_per_mm, line, confidence)

  2. Bug fix: fft_dominant_frequency returned only 2 values (None, None) on
     failure while the caller unpacks 3 -> it now returns (None, None, None).

  3. Bug fix: the "no group discovered" test now counts the REAL groups (it
     excludes the -1 "unclassified" label) instead of relying on the length of
     np.unique(gid).

The in-code comments of the original are kept (translated) where they document the
algorithm.
"""

import warnings

import numpy as np
from PIL import Image
from scipy.signal import find_peaks, hilbert

from ruler_confidence import row_spectrum, group_evidence, ruler_confidence


warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Parameters (kept from the original; the pipeline passes ratio/graduation in)
# --------------------------------------------------------------------------- #
GRADUATION_MM = 1.0

MIN_FREQ_RATIO = 0.005
MAX_FREQ_RATIO = 0.1
PEAK_PROMINENCE = 0.1

RATIO = 5
PHASE = 0

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


def observed_cycles(row, period_px, env_threshold=0.2):
    N = len(row)
    f0 = 1.0 / period_px
    row_c = row - row.mean()                 # NO window here (the edges are kept)

    fft = np.fft.rfft(row_c)
    freqs = np.fft.rfftfreq(N)
    bw = 0.5 * f0                            # +/-50 % around f0
    fft[(freqs < f0 - bw) | (freqs > f0 + bw)] = 0
    filtered = np.fft.irfft(fft, n=N)

    env = np.abs(hilbert(filtered))          # amplitude envelope
    threshold = env_threshold * env.max()

    peaks, _ = find_peaks(filtered, distance=max(1, period_px * 0.5))
    peaks = peaks[env[peaks] >= threshold]
    return len(peaks)


def _initial_slope(seed, period, lines, order, rank, delta, win=3):
    """Estimate the local slope period=f(lines) around the seed."""
    pos = rank[seed]
    window = 4.0 * delta
    idx = []
    for d in range(-win, win + 1):
        p = pos + d
        if 0 <= p < len(order):
            c = order[p]
            if abs(period[c] - period[seed]) <= window:
                idx.append(c)
    if len(idx) >= 2:
        a, _ = np.polyfit(lines[idx], period[idx], 1)
        return a
    return 0.0


def _grow(seed, direction, members, group_id, gid,
          period, lines, n_cycles, order, rank, delta, max_jumps, slope0):
    """Extend the group from the seed in one direction of the 'lines' axis."""
    pos = rank[seed]
    jumps = 0
    n = len(order)
    while True:
        pos += direction
        if pos < 0 or pos >= n:
            break
        cand = order[pos]

        if group_id[cand] != -1:             # candidate already assigned -> jump
            jumps += 1
            if jumps > max_jumps:
                break
            continue

        lm = lines[members]
        pm = period[members]
        cm = n_cycles[members]
        if len(members) >= 2:
            a, b = np.polyfit(lm, pm, 1)
            period_pred = a * lines[cand] + b
        else:
            period_pred = pm[0] + slope0 * (lines[cand] - lm[0])

        if len(members) >= 2:
            a, b = np.polyfit(lm, cm, 1)
            cycles_pred = a * lines[cand] + b
        else:
            cycles_pred = cm[0] + slope0 * (lines[cand] - lm[0])

        if abs(period[cand] - period_pred) <= delta:
            group_id[cand] = gid
            members.append(cand)
            jumps = 0
        else:
            jumps += 1
            if jumps > max_jumps:
                break


def find_groups(period, magnitude, lines, n_cycles, delta,
                max_jumps=2, n_groups=None, min_cycles=20):
    """Group the points (period vs lines) aligned on a same straight line.

    Returns: group_id (int array). Group 0 = most populated group after
    relabelling.
    """
    period = np.asarray(period, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    lines = np.asarray(lines, dtype=float)
    n_cycles = np.asarray(n_cycles, dtype=float)
    n = len(period)

    group_id = np.full(n, -1, dtype=int)     # -1 = free, -2 = rejected
    order = np.argsort(lines)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    gid = 0
    while np.any(group_id == -1):
        if n_groups is not None and gid >= n_groups:
            break
        free = np.where(group_id == -1)[0]
        seed = free[np.argmax(magnitude[free])]

        members = [seed]
        group_id[seed] = gid
        slope0 = _initial_slope(seed, period, lines, order, rank, delta)
        _grow(seed, +1, members, group_id, gid,
              period, lines, n_cycles, order, rank, delta, max_jumps, slope0)
        _grow(seed, -1, members, group_id, gid,
              period, lines, n_cycles, order, rank, delta, max_jumps, slope0)

        if n_cycles[members].mean() > min_cycles:
            gid += 1
        else:
            group_id[members] = -2

    group_id[group_id == -2] = -1

    # relabelling: group 0 = the most populated
    ids = [g for g in np.unique(group_id) if g != -1]
    sizes = {g: len(np.where(group_id == g)[0]) for g in ids}
    group_order = sorted(ids, key=lambda g: sizes[g], reverse=True)
    remap = {old: new for new, old in enumerate(group_order)}

    new_id = np.full_like(group_id, -1)
    for old, new in remap.items():
        new_id[group_id == old] = new
    return new_id


def gray_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Luma of an RGB uint8 image, matching load_image (R,G,B weights)."""
    return np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0


def detect_ruler(img_path, ratio=RATIO, phase=PHASE):
    """Path-based wrapper (kept for backward compatibility)."""
    base_img_gray, _base_img_color = load_image(img_path)
    return detect_ruler_from_gray(base_img_gray, ratio=ratio, phase=phase)


def detect_ruler_from_rgb(img_rgb: np.ndarray, ratio=RATIO, phase=PHASE):
    """Array-based entry point (RGB image already in memory)."""
    return detect_ruler_from_gray(gray_from_rgb(img_rgb), ratio=ratio, phase=phase)


def detect_ruler_from_gray(base_img_gray: np.ndarray, ratio=RATIO, phase=PHASE):
    """Detect a ruler and return (px_per_mm, line, confidence).

    confidence : ruler confidence in [0, 1], computed by
                 ruler_confidence.ruler_confidence(). Returns (None, None, None)
                 when no ruler frequency is found.

    This is the shared core: detect_ruler (path) and detect_ruler_from_rgb
    (in-memory image, used for Hugging Face) both funnel through here.
    """
    reduced_img_gray = base_img_gray[phase::ratio, phase::ratio]
    H, _W = reduced_img_gray.shape

    row_indices = np.arange(H)
    if N_LINES is not None and N_LINES < H:
        row_indices = np.linspace(0, H - 1, N_LINES, dtype=int)

    results, n_cycles, row_specs = [], [], []
    for i in row_indices:
        row = reduced_img_gray[i, :]
        rs = row_spectrum(row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE)
        if rs is not None:
            results.append((i, rs.period_px, rs.phase_rad, rs.magnitude))
            n_cycles.append(observed_cycles(row, rs.period_px))
            row_specs.append(rs)

    # No row produced a valid dominant frequency -> no ruler.
    if len(results) == 0:
        return None, None, None

    rows_arr = np.array([r[0] for r in results], dtype=float)
    phases_arr = np.array([r[2] for r in results], dtype=float)
    mag_arr = np.array([r[3] for r in results], dtype=float)
    periods_arr = np.array([r[1] for r in results], dtype=float)

    gid = find_groups(periods_arr, mag_arr, rows_arr, n_cycles,
                      delta=0.2, n_groups=5)

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

    ev = group_evidence(gid, row_specs, rows_arr, periods_arr,
                            reduced_img_gray, n_candidate_rows=len(row_indices))
    conf, debug = ruler_confidence(ev)

    return px_per_mm, max_line_idx * ratio, conf
