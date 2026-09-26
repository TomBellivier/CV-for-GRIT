"""
ruler_confidence.py
===================

Confidence on the presence of a ruler in the image.

The problem of the old version
------------------------------
The old function compared the groups with each other: it answered "WHICH group is
the ruler?", not "IS THERE a ruler?". When the image contains no ruler, the detector
often finds a single spurious group, and the old rule then returned 1.0: the maximum
confidence on the worst case.

Here the two questions are separated:

    confidence = presence x uniqueness

  * presence   : is there really a ruler? (new, 3 absolute criteria)
  * uniqueness : was the right group chosen? (the original logic)

The 3 presence criteria are RATIOS: they depend neither on the width of the image nor
on its contrast, so they are comparable from one image to another.

Wiring in ruler_detection.py
----------------------------
In detect_ruler_from_gray, replace the loop over the rows with:

    from ruler_confidence import row_spectrum, group_evidence, ruler_confidence

    results, n_cycles, row_specs = [], [], []
    for i in row_indices:
        row = reduced_img_gray[i, :]
        rs = row_spectrum(row, MIN_FREQ_RATIO, MAX_FREQ_RATIO, PEAK_PROMINENCE)
        if rs is not None:
            results.append((i, rs.period_px, rs.phase_rad, rs.magnitude))
            n_cycles.append(observed_cycles(row, rs.period_px))
            row_specs.append(rs)

then, after computing gid:

    ev = group_evidence(gid, row_specs, rows_arr, periods_arr,
                        reduced_img_gray, n_candidate_rows=len(row_indices))
    conf, debug = ruler_confidence(ev)

row_spectrum exposes .period_px / .phase_rad / .magnitude: the rest of the pipeline
(find_groups, px_per_mm) is unchanged.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
# Thresholds: (pivot value, width of the doubt zone).
# Plausible starting values, TO RECALIBRATE on your images -> see calibrate().
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "snr_db":          (6.3617, 1.7858),    # sharpness of the peak
    "phase_coherence": (0.8548, 0.2485),   # alignment of the ruler
    "support_frac":    (0.0281, 0.0055),   # fraction of rows involved
}


# --------------------------------------------------------------------------- #
# Analysis of one row
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=16)
def _hann(n):
    """Cached Hann window (it only depends on the length)."""
    w = np.hanning(n)
    w.flags.writeable = False
    return w


@lru_cache(maxsize=16)
def _freqs(n):
    """Cached frequency axis."""
    f = np.fft.rfftfreq(n)
    f.flags.writeable = False
    return f


@lru_cache(maxsize=16)
def _median_windows(n, stride=8):
    """Indices of the sliding-median windows, cached.

    These indices only depend on the length of the spectrum: they are therefore
    identical for every row of a same image, and computed once.
    Clipping at the bounds replaces the "edge" padding without allocating anything.
    """
    k = max(11, n // 16)
    if k % 2 == 0:
        k += 1
    half = k // 2
    centres = np.arange(0, n, stride)
    if centres[-1] != n - 1:
        centres = np.append(centres, n - 1)
    idx = np.clip(centres[:, None] + np.arange(-half, half + 1)[None, :],
                  0, n - 1)
    return idx, half, centres.astype(float), np.arange(n, dtype=float)


def _noise_floor(mag):
    """Local noise level of the spectrum (sliding median, subsampled).

    Image rows have a 1/f spectrum: without this, the low frequencies crush
    everything. Dividing by this floor makes the peak comparable between images.

    The median is only computed at one position out of 8, then interpolated: the
    floor varies slowly with the frequency, so the result is indistinguishable from a
    full sliding median, for much less computation.
    """
    idx, half, centres, grid = _median_windows(len(mag))
    med = np.partition(mag[idx], half, axis=1)[:, half]
    return np.interp(grid, centres, med) + 1e-12


@dataclass
class RowSpectrum:
    """Analysis of one row. The first 3 fields replace the old tuple."""
    period_px: float
    phase_rad: float
    magnitude: float     # raw magnitude, for find_groups (unchanged)
    snr_db: float        # peak / local noise, in dB -> dimensionless
    freq: float          # peak frequency (cycles/pixel)


def row_spectrum(row, min_period_ratio, max_period_ratio, prominence,
                 whiten=True):
    """FFT of a row -> RowSpectrum, or None if no valid peak.

    Replaces fft_dominant_frequency, adding snr_db.
    whiten=True searches the peak on the spectrum rid of the 1/f background.
    """
    N = len(row)
    if N < 32:
        return None

    x = (row - row.mean()) * _hann(N)
    X = np.fft.rfft(x)
    freqs = _freqs(N)
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


def phases_at(rows_pixels, f):
    """Phases of SEVERAL rows at a same frequency f, in a single product.

    Necessary because from one row to another the peak can fall into a different
    bin: every phase of the group must be measured at the SAME frequency, otherwise
    the measured gap means nothing.

    rows_pixels: array (n_rows, N). The complex exponential is built only once and
    shared, instead of being rebuilt for each row.
    """
    a = np.atleast_2d(np.asarray(rows_pixels, dtype=float))
    N = a.shape[1]
    x = (a - a.mean(axis=1, keepdims=True)) * _hann(N)
    e = np.exp(-2j * np.pi * f * np.arange(N))
    return np.angle(x @ e)


def phase_at(row, f):
    """Phase of a single row (kept for compatibility)."""
    return float(phases_at(row, f)[0])


def phase_coherence(rows, phases, n_slopes=1024):
    """Alignment of the ruler, in [0, 1]. This is the strongest criterion.

    A tilted ruler gives a phase that advances REGULARLY from one row to the next
    (straight line -> constant phase). A natural texture (grass, fabric, ripples)
    gives phases in disorder.

    Every possible slope is tested and the best one is kept, then the score that
    would be obtained by pure chance is subtracted (otherwise a slope that "works"
    is always found with few rows).

    Testing every slope is exactly an FFT of the phases placed on the row axis: the
    same slopes are obtained, in O(M log M) instead of O(M x n_rows).
    """
    n = len(rows)
    if n < 6:
        return 0.0
    r = np.asarray(rows, dtype=float)
    r = np.round(r - r.min()).astype(int)
    if r.max() <= 0:
        return 0.0

    # FFT size: power of 2, at least the extent of the rows and the requested slope
    # resolution (no aliasing possible)
    M = 1 << int(max(n_slopes, r.max() + 1) - 1).bit_length()

    grid = np.zeros(M, dtype=complex)
    np.add.at(grid, r, np.exp(1j * np.asarray(phases, dtype=float)))
    best = float(np.abs(np.fft.fft(grid)).max() / n)

    chance = float(np.sqrt(np.log(max(n, 2)) / n))
    if chance >= 1.0:
        return 0.0
    return float(np.clip((best - chance) / (1.0 - chance), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Aggregation per group
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """What was measured on the main group."""
    snr_db: float = 0.0
    phase_coherence: float = 0.0
    support_frac: float = 0.0
    group_snr_db: list = field(default_factory=list)  # [main, secondaries]
    n_rows: int = 0


def group_evidence(gid, row_specs, rows_arr, periods_arr, reduced_img_gray,
                   n_candidate_rows):
    """Measure the 3 criteria on the main group (gid == 0).

    n_candidate_rows: number of rows ANALYSED (not only kept).
    """
    ev = Evidence()
    gid = np.asarray(gid)
    main = np.where(gid == 0)[0]
    if len(main) == 0:
        return ev

    rows_main = np.asarray(rows_arr, dtype=float)[main]
    periods_main = np.asarray(periods_arr, dtype=float)[main]
    ev.n_rows = len(main)

    # 1. sharpness of the peak
    ev.snr_db = float(np.median([row_specs[i].snr_db for i in main]))

    # 2. fraction of rows involved
    ev.support_frac = len(main) / max(n_candidate_rows, 1)

    # 3. alignment, measured at a reference frequency shared by the group
    f_ref = 1.0 / float(np.median(periods_main))
    phases = phases_at(reduced_img_gray[rows_main.astype(int), :], f_ref)
    ev.phase_coherence = phase_coherence(rows_main, phases)

    # context: level of each group, for the uniqueness
    for g in sorted(int(g) for g in np.unique(gid) if g != -1):
        idx = np.where(gid == g)[0]
        ev.group_snr_db.append(
            float(np.median([row_specs[i].snr_db for i in idx])))
    return ev


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def _score(value, x0, width):
    """Turn a measure into a score between 0 and 1 (smooth transition at x0)."""
    return float(1.0 / (1.0 + np.exp(-(value - x0) / width)))


def ruler_confidence(ev):
    """Final confidence in [0, 1], + a debug dict to log per image.

    presence = geometric mean of the 3 scores. Geometric and not arithmetic: a real
    ruler must satisfy the 3 criteria, so a zero score must bring everything down,
    and not be compensated by the two others.
    """
    if ev is None or ev.n_rows == 0:
        return 0.0, {"reason": "no group"}

    measures = {
        "snr_db": ev.snr_db,
        "phase_coherence": ev.phase_coherence,
        "support_frac": ev.support_frac,
    }
    scores = {k: max(_score(v, *THRESHOLDS[k]), 1e-3) for k, v in measures.items()}

    presence = float(np.prod(list(scores.values())) ** (1.0 / len(scores)))
    uniqueness = _uniqueness(ev.group_snr_db)
    conf = float(np.clip(presence * uniqueness, 0.0, 1.0))

    debug = {
        "confidence": conf,
        "presence": presence,
        "uniqueness": uniqueness,
        "n_groups": len(ev.group_snr_db),
        "n_rows": ev.n_rows,
        "measures": measures,
        "scores": scores,
    }
    return conf, debug


def _uniqueness(group_snr_db):
    """The original logic, but on the SNR (dimensionless).

    A single group -> 1.0 is correct HERE: nothing competes with the kept frequency.
    What was wrong was to deduce from it that a ruler exists.
    """
    m = np.asarray(list(group_snr_db), dtype=float)
    if m.size <= 1:
        return 1.0
    if not np.isfinite(m[0]) or m[0] <= 0:
        return 0.0
    return float(np.clip(((m[0] - m[1:]) / m[0]).clip(0, 1).mean(), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Calibration on annotations (to run BY HAND, not in the pipeline)
# --------------------------------------------------------------------------- #
def _evidence_from_path(img_path, ratio):
    """Replay the full pipeline on an image -> (Evidence, median_cycles).

    Returns (None, 0.0) if no group is found (= the detector already says "nothing
    here", which is a valid answer that no threshold will change).
    """
    from ruler_detection import (MIN_FREQ_RATIO, MAX_FREQ_RATIO,
                                 PEAK_PROMINENCE, load_image,
                                 observed_cycles, find_groups)

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
            cycles.append(observed_cycles(row, rs.period_px))

    if not specs:
        return None, 0.0

    rows = np.asarray(rows, dtype=float)
    periods = np.asarray(periods, dtype=float)
    gid = find_groups(periods, np.asarray(mags), rows, cycles,
                      delta=0.2, n_groups=5)

    if not np.any(gid == 0):
        return None, 0.0

    ev = group_evidence(gid, specs, rows, periods, reduced, n_candidate_rows=H)
    cyc_main = float(np.median(np.asarray(cycles, dtype=float)[gid == 0]))
    return ev, cyc_main


def calibrate_from_annotations(json_path, images_root=None, ratio=5,
                               positive_label=2, verbose=True):
    """Compute the optimal thresholds from annotation.json. MANUAL USE.

    annotation.json: {"path/image.jpg": "2", ...}
        0 = nothing, 1 = scale bar, 2 = ruler.

    The POSITIVE class is class 2 (ruler). Classes 0 AND 1 are negatives: a scale bar
    must not be read as a ruler. The report gives the AUC separately against each
    class, to see which one resists.

    Returns (thresholds, report):
      * thresholds : dict ready to copy into THRESHOLDS
      * report     : diagnostics per criterion + performance of the final confidence

    Example:
        th, rep = calibrate_from_annotations("annotation.json", images_root="data")
        print(rep["decision"])
    """
    import json
    from pathlib import Path

    from scipy.stats import rankdata

    root = Path(images_root) if images_root else None
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # --- 1. feature extraction -------------------------------------------------
    data = {0: [], 1: [], 2: []}      # label -> list of (Evidence, cycles)
    n_without_group = {0: 0, 1: 0, 2: 0}
    n_unreadable = 0

    for rel_path, label in annotations.items():
        try:
            label = int(label)
        except (TypeError, ValueError):
            continue
        if label not in data:
            continue

        path = root / rel_path if root else Path("../../..") / Path(str(rel_path).replace("\\", "/")[6:])
        path  = path.resolve()
        try :
            ev, cyc = _evidence_from_path(path, ratio)
        except Exception as err:                      # missing/corrupted image
            n_unreadable += 1
            if verbose:
                print(f"  skipped ({err}): {path}")
            continue

        if ev is None:
            n_without_group[label] += 1
        else:
            data[label].append((ev, cyc))

        if verbose and (sum(len(v) for v in data.values()) % 25 == 0):
            print(f"  ... {sum(len(v) for v in data.values())} images processed")

    pos = [e for e, _ in data[positive_label]]
    neg_per_class = {lab: [e for e, _ in data[lab]]
                     for lab in data if lab != positive_label}
    neg = [e for lst in neg_per_class.values() for e in lst]

    print(n_unreadable, "unreadable images")

    if len(pos) < 5 or len(neg) < 5:
        raise ValueError(
            f"Not enough usable images: {len(pos)} positives, "
            f"{len(neg)} negatives. Check the paths of the json.")

    # --- 2. separation tools ---------------------------------------------------
    def auc(a, b):
        """Area under the ROC curve (Mann-Whitney). 0.5 = useless, 1 = perfect."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.size == 0 or b.size == 0:
            return float("nan")
        r = rankdata(np.concatenate([a, b]))
        return float((r[:a.size].sum() - a.size * (a.size + 1) / 2)
                     / (a.size * b.size))

    def best_threshold(a, b):
        """Threshold maximising (true positive rate - false positive rate)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        vals = np.unique(np.concatenate([a, b]))
        if vals.size < 2:
            return float(vals[0]) if vals.size else 0.0, 0.0
        cands = (vals[:-1] + vals[1:]) / 2.0
        j = (a[:, None] >= cands).mean(0) - (b[:, None] >= cands).mean(0)
        k = int(np.argmax(j))
        return float(cands[k]), float(j[k])

    # --- 3. one threshold per criterion ------------------------------------------
    thresholds, criteria_report = {}, {}
    for name in THRESHOLDS:
        v_pos = np.array([getattr(e, name) for e in pos])
        v_neg = np.array([getattr(e, name) for e in neg])

        x0, j = best_threshold(v_pos, v_neg)

        # width = extent of the doubt zone between the two classes.
        # well separated classes -> positive gap -> sharp transition
        # overlapping classes -> negative gap -> smoother transition
        gap = abs(float(np.quantile(v_pos, 0.10) - np.quantile(v_neg, 0.90)))
        extent = float(np.ptp(np.concatenate([v_pos, v_neg])))
        width = max(gap / 4.0, 0.05 * extent, 1e-3)

        thresholds[name] = (round(x0, 4), round(width, 4))
        criteria_report[name] = {
            "x0": round(x0, 4),
            "width": round(width, 4),
            "youden_J": round(j, 3),
            "auc_vs_all": round(auc(v_pos, v_neg), 3),
            "auc_vs_nothing": round(
                auc(v_pos, [getattr(e, name) for e in neg_per_class.get(0, [])]), 3),
            "auc_vs_scalebar": round(
                auc(v_pos, [getattr(e, name) for e in neg_per_class.get(1, [])]), 3),
            "median_ruler": round(float(np.median(v_pos)), 3),
            "median_scalebar": round(float(np.median(
                [getattr(e, name) for e in neg_per_class.get(1, [])] or [np.nan])), 3),
            "median_nothing": round(float(np.median(
                [getattr(e, name) for e in neg_per_class.get(0, [])] or [np.nan])), 3),
        }

    # --- 4. candidate criterion not kept: the number of graduations --------------
    # It is the one that best separates a ruler (many graduations) from a scale bar
    # (a few ticks). If it comes out better than the 3 current criteria against
    # class 1, it must be added to THRESHOLDS.
    cyc = {lab: np.array([c for _, c in data[lab]]) for lab in data}
    cycles_candidate = {
        "auc_vs_scalebar": round(auc(cyc[positive_label], cyc.get(1, [])), 3),
        "auc_vs_nothing": round(auc(cyc[positive_label], cyc.get(0, [])), 3),
        "median_ruler": round(float(np.median(cyc[positive_label])), 1),
        "median_scalebar": round(float(np.median(cyc[1])), 1) if len(cyc[1]) else None,
    }

    # --- 5. performance of the final confidence with these thresholds ------------
    def conf_with(ev, th):
        scores = [max(_score(getattr(ev, n), *th[n]), 1e-3) for n in th]
        presence = float(np.prod(scores) ** (1.0 / len(scores)))
        return presence * _uniqueness(ev.group_snr_db)

    c_pos = np.array([conf_with(e, thresholds) for e in pos])
    c_neg = np.array([conf_with(e, thresholds) for e in neg])
    decision_threshold, _ = best_threshold(c_pos, c_neg)

    # the images without a group count as confidence 0
    n_pos_tot = len(pos) + n_without_group[positive_label]
    n_neg_tot = len(neg) + sum(n_without_group[l] for l in n_without_group
                               if l != positive_label)
    recall = float((c_pos >= decision_threshold).sum()) / max(n_pos_tot, 1)
    false_pos = float((c_neg >= decision_threshold).sum()) / max(n_neg_tot, 1)

    report = {
        "n_images": {"ruler": n_pos_tot, "scalebar": len(neg_per_class.get(1, [])),
                     "nothing": len(neg_per_class.get(0, [])),
                     "unreadable": n_unreadable},
        "without_group": n_without_group,
        "criteria": criteria_report,
        "candidate_n_graduations": cycles_candidate,
        "decision": {
            "confidence_threshold": round(decision_threshold, 3),
            "ruler_recall": round(recall, 3),
            "false_positive_rate": round(false_pos, 3),
            "confidence_auc": round(auc(c_pos, c_neg), 3),
        },
    }

    if verbose:
        print("\n--- proposed thresholds (to copy into THRESHOLDS) ---")
        for k, v in thresholds.items():
            print(f'    "{k}": {v},   # AUC vs scalebar = '
                  f'{criteria_report[k]["auc_vs_scalebar"]}')
        print(f"\ndecision threshold on the confidence: "
              f"{report['decision']['confidence_threshold']}")
        print(f"recall {report['decision']['ruler_recall']}  |  "
              f"false positives {report['decision']['false_positive_rate']}")

    return thresholds, report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Calibration of the confidence on the presence of a ruler")
    parser.add_argument("--json", default = "../annotations.json", help="annotation.json")
    parser.add_argument("--images_root", help="root of the images (optional)")
    parser.add_argument("--ratio", type=int, default=5,
                        help="image reduction for the FFT computation")
    args = parser.parse_args()

    calibrate_from_annotations(args.json, images_root=args.images_root,
                               ratio=args.ratio, verbose=True)
