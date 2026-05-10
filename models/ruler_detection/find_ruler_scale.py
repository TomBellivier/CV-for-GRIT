import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.stats import circmean
import pandas as pd
from PIL import Image
from tqdm import tqdm
import warnings
from pathlib import Path
from math import ceil
warnings.filterwarnings('ignore')

GRADUATION_MM = 1.0
MIN_PERIOD_RATIO = 0.005
MAX_PERIOD_RATIO = 0.1
RULER_RATIO = 0

def fft_dominant_frequency(row: np.ndarray):
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
    (période_px, phase_rad, max_magnitude) ou (None, None, None) si aucun pic valide
    """
    N = len(row)

    row_centered = row - row.mean()

    window = np.hanning(N)
    row_windowed = row_centered * window

    fft_vals = np.fft.rfft(row_windowed)
    freqs    = np.fft.rfftfreq(N)   # fréquences normalisées [0, 0.5]

    magnitude = np.abs(fft_vals)

    f_min = 1.0 / (MAX_PERIOD_RATIO * len(row))
    f_max = 1.0 / (MIN_PERIOD_RATIO * len(row))

    mask = (freqs >= f_min) & (freqs <= f_max)
    if mask.sum() == 0:
        return None, None, None

    mag_masked = magnitude.copy()
    mag_masked[~mask] = 0

    peaks, props = find_peaks(mag_masked)
                             # prominence=prominence * mag_masked.max())
    if len(peaks) == 0:
        return None, None, None

    best_peak = peaks[np.argmax(mag_masked[peaks])]

    period_px = 1.0 / freqs[best_peak]
    phase_rad = np.angle(fft_vals[best_peak])

    return period_px, phase_rad, max(mag_masked[peaks])

def correct_period(results, ax=None):

    periods_arr = np.array([r[1] for r in results], dtype=float)
    rows_arr   = np.array([r[0] for r in results], dtype=float)
    mag_arr = np.array([r[3] for r in results], dtype=float)
    mag_arr = np.array([r[3] for r in results], dtype=float)

    max_mag = max(mag_arr)
    filtered_rows = []
    filtered_phases = []
    filtered_periods = []
    for i in range(len(results)):
        if abs(results[i][3] - max_mag) < max_mag * 0.01:  # Use a small tolerance for floating-point comparison
            filtered_rows.append(results[i][0])
            filtered_phases.append(results[i][2])
            filtered_periods.append(results[i][1])
    filtered_rows_arr = np.array(filtered_rows, dtype=float)
    filtered_phases_arr = np.array(filtered_phases, dtype=float)
    filtered_periods_arr = np.array(filtered_periods, dtype=float)
    print(len(filtered_rows_arr), len(filtered_phases_arr), len(filtered_periods_arr))

    T_median = np.median(filtered_periods_arr)

    phases_unwrapped = np.unwrap(filtered_phases_arr)

    slope_phase, intercept_phase = np.polyfit(filtered_rows_arr, phases_unwrapped, 1)

    tan_theta = slope_phase * T_median / (2 * np.pi)
    theta_rad = np.arctan(tan_theta)

    if ax is not None:
        ax.scatter(rows_arr, periods_arr, s=4, alpha=0.5, c=[res[3] for res in results], cmap='viridis',)
        ax.axhline(T_median, color='red', ls='--', lw=1.5,
                        label=f'Median = {T_median:.2f} px')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    print(max_mag)

    return T_median * np.cos(theta_rad), theta_rad

def load_image(path: str, detected_box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    detection box : [x1, y1, x2, y2] in pixel coordinates, or None if no box
    """
    img = Image.open(path).convert("RGB")
    img_color = np.array(img)

    img_gray = np.dot(img_color[..., :3], [0.299, 0.587, 0.114]).astype(np.float32) / 255.0

    # replace every pixel in the detection box with the mean of the surrounding pixels
    if detected_box is not None:
        x1, y1, x2, y2 = int(detected_box[0]), int(detected_box[1]), int(detected_box[2]), int(detected_box[3])
        surrounding_pixels = []
        if y1 > 0:
            surrounding_pixels.append(img_gray[y1-1, x1:x2])
        if y2 < img_gray.shape[0]:
            surrounding_pixels.append(img_gray[y2, x1:x2])
        if x1 > 0:
            surrounding_pixels.append(img_gray[y1:y2, x1-1])
        if x2 < img_gray.shape[1]:
            surrounding_pixels.append(img_gray[y1:y2, x2])
        if len(surrounding_pixels) > 0:
            mean_value = np.mean(surrounding_pixels)
            img_gray[y1:y2, x1:x2] = mean_value

    return img_gray, img_color

def estimate_scale(image_path, detected_box=None, ax=None):
    img_gray, img_color = load_image(image_path, detected_box=detected_box)
    # img_arr = img_gray[int(0.75*img_gray.shape[0]):, :]
    img_arr = img_gray[int(RULER_RATIO*img_gray.shape[0]):, :]

    results = []
    for i in range(img_arr.shape[0]):
        period_px, phase_rad, mag = fft_dominant_frequency(img_arr[i])
        if period_px is not None:
            results.append((i, period_px, phase_rad, mag))

    if len(results) == 0:
        raise ValueError("No image rows with valid periodicity found.")

    T_corrected, angle = correct_period(results, ax=ax)

    if T_corrected > (MAX_PERIOD_RATIO * img_arr.shape[1] * 0.5):
        print(f"Warning: Detected period {T_corrected:.2f} px is close to the maximum expected. Results may be unreliable.")


    scale_mm_per_px = T_corrected / GRADUATION_MM

    return scale_mm_per_px, angle

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def estimate_directory_scale(dir_path, csv=None):
    import os
    scales = []
    angles = []
    all_files = list(Path(dir_path).rglob('*.png')) + list(Path(dir_path).rglob('*.jpg'))
    if csv is not None:
        all_files = [csv['directory'].iloc[i] + "/" +csv['filename'].iloc[i] for i in range(len(csv))]

    grid_size = ceil(len(all_files)**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(50, 50))
    i = 0
    for filename in tqdm(all_files):
        print(filename)
        image_path = os.path.join(dir_path, filename).replace("\\", "/")
        scale, angle = estimate_scale(image_path, detected_box=None, ax=axes[i%grid_size, i//grid_size] if len(axes) > 0 else None)
        axes[i%grid_size, i//grid_size].set_title(f"{filename}")
        scales.append(scale)
        angles.append(angle)
        i += 1
    plt.tight_layout()
    plt.show()
    return scales, angles

def evaluate_model(predicted_scales, true_scales):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    mae = mean_absolute_error(true_scales, predicted_scales)
    rmse = np.sqrt(mean_squared_error(true_scales, predicted_scales))
    return mae, rmse

if __name__ == "__main__":
    csv_path = "C:/Users/tombe/Documents/_MLE/CV-for-GRIT/databases/luomus_ruler_scale.csv"
    dir_path = "D:/luomus_leps_pictures"
    csv_data = load_csv(csv_path)
    predicted_scales, predicted_angles = estimate_directory_scale(dir_path, csv_data)
    print(predicted_scales, predicted_angles)
    true_scales = csv_data['final_px_per_mm'].values
    mae, rmse = evaluate_model(predicted_scales, true_scales)
    print(f"MAE: {mae:.4f} mm/px")
    print(f"RMSE: {rmse:.4f} mm/px")

    # add the predicted values and errors in the csv file
    csv_data['predicted_px_per_mm'] = predicted_scales
    csv_data['angle_rad'] = predicted_angles
    csv_data['error_mm_per_px'] = np.abs(csv_data['final_px_per_mm'] - csv_data['predicted_px_per_mm'])
    csv_data.to_csv("C:/Users/tombe/Documents/_MLE/CV-for-GRIT/databases/luomus_ruler_scale_with_predictions.csv", index=False)
